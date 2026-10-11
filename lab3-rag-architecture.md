# Lab 3 — RAG System Architecture

**Assumed context (carried from Labs 1–2):** 7B model served via vLLM on A100 80GB, ~10M-document corpus, mixed sources (PDF, HTML, tickets, DB), access-controlled. Conceptual design — no implementation.

## 1. Architecture Diagram

```
                        INGESTION  (offline, batch, re-runnable)

 ┌──────────┐   ┌────────────┐   ┌────────────┐   ┌──────────┐   ┌────────────┐
 │ Sources  │──▶│ Parse &    │──▶│ Chunk      │──▶│ Embed    │──▶│ Index      │
 │ PDF/HTML │   │ normalize  │   │ structure- │   │ normalize│   │ HNSW/IVF   │
 │ tickets  │   │ layout +   │   │ aware,     │   │ vectors  │   │ + metadata │
 │ DB/docs  │   │ metadata   │   │ 256-512 tok│   │          │   │ + versions │
 └──────────┘   └────────────┘   └────────────┘   └──────────┘   └─────┬──────┘
   ACL · doc version · timestamp · header path · title travel as metadata┘
                                                                        │
                        QUERY  (online, latency-bound)                  ▼
 ┌───────┐  ┌───────────┐  ┌────────────────┐  ┌───────────┐  ┌──────────────┐
 │ User  │─▶│ Rewrite / │─▶│ Hybrid search  │─▶│ Rerank    │─▶│ Assemble     │
 │ query │  │ decompose │  │ dense ⊕ BM25   │  │ cross-enc │  │ context      │
 │       │  │ (optional)│  │ RRF fusion     │  │ 50 → 5    │  │ token budget │
 └───────┘  └───────────┘  │ ACL PRE-filter │  └───────────┘  │ + dedup/MMR  │
                           └────────────────┘                 └──────┬───────┘
                                                                     │
                                              ┌──────────────────────▼──────────┐
                                              │ LLM (vLLM, temp 0-0.3)          │
                                              │ "Answer only from context.      │
                                              │  Cite chunk IDs. If unanswerable│
                                              │  say so."                       │
                                              └──────────────────────┬──────────┘
                                                                     │
                        ┌────────────────────────────────────────────▼─────────┐
                        │ Answer + citations                                   │
                        │   ├─ citation verifier (does chunk N support claim?) │
                        │   ├─ groundedness score (LLM/NLI judge)              │
                        │   └─ full trace → log store → golden-set eval + alerts│
                        └──────────────────────────────────────────────────────┘
```

## 2. Ingestion

**Parsing is not text extraction.** Layout-aware parsing preserves tables, headers, code blocks, and list structure; naive `pdftotext` destroys tables and merges columns, producing chunks that are unretrievable *and* unreadable by the LLM. Capture metadata at parse time — it cannot be reconstructed later: source, title, header path, doc version hash, timestamp, and **ACL**.

**Chunking is the highest-leverage decision in the system.** Tradeoff: small chunks embed precisely (one vector, one topic) but fragment context; large chunks are coherent but dilute the embedding into a single vector summarizing many topics, which discriminates poorly.

| Strategy | Notes |
| --- | --- |
| Fixed-size + overlap | Simple baseline; splits sentences and tables arbitrarily |
| **Recursive / structure-aware** | Falls back paragraph → sentence → token; respects document hierarchy. Default choice |
| Semantic chunking | Splits where inter-sentence similarity drops; adapts to topic shifts, costs an embedding pass |
| **Parent-child (small-to-big)** | Embed small chunks for precise retrieval, return the *parent* section for coherent context. Resolves the tradeoff above |
| Contextual enrichment | Prepend LLM-generated chunk-in-document context before embedding; large precision gain, large ingestion cost |

Prepend the header path and doc title to each chunk's text before embedding, so provenance is encoded in the vector rather than only in metadata.

**Embeddings:** normalize vectors so cosine reduces to a dot product. Matryoshka models allow dimension truncation (3072 → 1024) for ~3× less index memory at modest quality cost. Fine-tuning the embedding model on contrastive pairs mined from real query/click logs beats any amount of index tuning — do that before buying a bigger vector DB. **Version the index by embedding-model ID**: changing models requires re-embedding the entire corpus, and without versioning you cannot roll back.

## 3. Storage and Index

| Index | Recall/latency | Memory | Use when |
| --- | --- | --- | --- |
| Flat (exact) | Perfect recall, O(N) | 1× | < ~1M vectors |
| IVF (+PQ) | Tunable via `nprobe` | Low with PQ | 10M–1B, bulk, trained once |
| **HNSW** | Best recall/latency | **1.5–2×** (graph) | Default; incremental inserts, no training |
| Quantized (SQ/PQ/binary) | Coarse | 4–32× less | Large scale — then rerank with full-precision vectors |

**pgvector is the underrated default.** If the corpus is under ~10M vectors and you already run Postgres, you get vector search, metadata filters, transactions, and backups in one system with no new operational surface. Move to a dedicated engine (Qdrant, Milvus, Weaviate) only when scale, filter complexity, or ingestion rate forces it.

**Filtering is a security boundary, not a performance detail.** ACL checks **must be pre-filter** (applied during the search). Post-filtering retrieves top-k then drops unauthorized hits — which leaks document existence through timing and result counts, and can silently return fewer than k results. Pre-filtering can degrade ANN graph traversal, so test recall *with* filters applied, not without.

## 4. Retrieval

**Hybrid is the standard, not an optimization.** Dense embeddings systematically miss exact identifiers — error codes, SKUs, function names, acronyms, proper nouns — precisely the tokens a technical query hinges on. BM25 catches them. Fuse with **Reciprocal Rank Fusion** (`score = Σ 1/(k + rank_i)`, k≈60), which combines ranked lists without needing to normalize incompatible score scales.

**Reranking is the largest precision gain per unit of effort.** A bi-encoder compresses query and chunk into independent vectors; a cross-encoder attends over both jointly and is far more accurate but too slow for the full corpus. So: retrieve top 50–100 cheaply, rerank, keep top 5. Costs 20–80 ms.

**Query transformation** (HyDE, decomposition, rewriting) helps multi-hop and vague queries but adds an LLM call per query — real latency and cost on the critical path. Make it conditional on detected query complexity rather than unconditional.

**Context assembly:**
- **Lost-in-the-middle:** LLMs attend best to the start and end of the context. Place the strongest chunk first or last, not buried in the middle.
- **Do not stuff the window.** Irrelevant chunks are active distractors, not neutral padding. Measure the marginal value of each additional chunk and stop where it flattens.
- **MMR** to trade relevance against redundancy when top-k chunks overlap heavily.
- **Prompt ordering matters for cost:** put the *static* system prompt and tool definitions first, retrieved chunks after. Prefix caching (Lab 2 §4) then hits on every request; reversing the order destroys the cache and makes every query pay full prefill.

## 5. Generation

Answer only from provided context; cite chunk IDs; explicitly permit "not in the provided documents." Temperature 0–0.3 for factual QA. The permission to refuse is the primary hallucination control — a model never allowed to say "I don't know" will invent an answer instead.

**Retrieval-confidence gating:** if the top reranker score falls below a threshold, refuse rather than generate from weak evidence. Monitor that refusal rate; a rising rate signals a content gap or corpus drift.

## 6. Evaluation — Two Stages, Measured Separately

An end-to-end score cannot tell you whether retrieval or generation failed. Instrument both.

- **Retrieval:** Recall@k, MRR, nDCG, context precision against labeled (query → relevant chunk) pairs. Mine labels from real queries + human review. LLM-synthesized questions (generate a question from a chunk, use that chunk as ground truth) are cheap to scale but produce unrealistically clean questions — treat as a smoke test, not a benchmark.
- **Generation:** faithfulness/groundedness (is every atomic claim entailed by the retrieved context?), answer relevance, context utilization. Via RAGAS/TruLens or LLM-as-judge with an explicit rubric.
- **LLM-as-judge caveats:** position bias, verbosity bias, self-preference. Pin the judge version, randomize option order, and **report judge–human agreement on a calibration subset**. If the judge silently degrades, every downstream metric is fiction.
- **Gate:** a versioned golden set of 200–500 real queries, run on every change to prompt, model, index, or embedding, with regression thresholds in CI.

## 7. Failure Modes

| Failure | Symptom | Fix |
| --- | --- | --- |
| Retrieval miss | Correct doc exists, not retrieved | Hybrid search, reranking, chunking change, embedding fine-tune |
| Generation miss | Correct doc retrieved, ignored or contradicted | Prompt contract, context reordering, larger/better model |
| **Stale index** | Confidently wrong, outdated answers | Doc-version hashing, incremental re-index, TTL, staleness alerts |
| Permission leak | User sees unauthorized content | Pre-filter ACL; test recall *with* filters |
| Unanswerable query answered | Plausible fabrication | Confidence gating, refusal permission, "not found" rate monitoring |
| Context stuffing | Rising cost, falling accuracy | Token budget, measure marginal value of k |
| **Prompt injection** | Retrieved text hijacks instructions | No reliable fix — treat retrieved content as untrusted data, never as instructions; privilege separation; output filtering |

**Next action:** build the golden set first, before any index tuning. Fifty real queries with human-labeled relevant chunks gives you stage-wise Recall@k and groundedness baselines; without it every architectural choice above is unfalsifiable and you will optimize whichever component is easiest to measure rather than whichever is broken.
