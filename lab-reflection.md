# Final Reflection — LLMOps vs Classical MLOps

## 1. What makes LLM Ops harder than classical ML Ops?

**There is no label.** Classical ML has a ground-truth target and a closed output space: AUC, F1, MAE are computable, and a model's output is unit-testable. An LLM emits unbounded text with no reference answer, so accuracy is not measurable — it is *judged*, by rubric, by human, or by another model. **You end up evaluating a model with a model**, which means your metric has its own version, bias, and failure modes and needs calibration of its own. That single change propagates everywhere: regression tests become distributional instead of exact-match, CI needs an eval harness, and "did this change make it better?" is a harder question than it was.

**Fluent failure.** A wrong classifier usually looks wrong — low confidence, implausible label. A wrong LLM writes a confident, well-formatted, plausible paragraph. The quality signal users perceive (fluency) is orthogonal to the signal that matters (correctness), so failure is *adversarial to detection*. Nothing in the output tells you to look closer.

**Non-determinism.** Sampling, batching-dependent numerics, and long context mean identical inputs give different outputs. You cannot assert equality; you must sample N times and compare distributions. Self-consistency across samples becomes a usable uncertainty proxy precisely because determinism is unavailable.

**The model is a dependency you do not control.** A closed-API model can be silently updated or deprecated behind a pinned alias, breaking your product with zero code changes on your side. A scikit-learn model you trained is frozen; this one is not. Requires version pinning, output contract tests, and a re-evaluation gate on any provider change.

**Prompts are unversioned natural-language code.** Behavior lives in English text edited by non-engineers, with no compiler, no type system, and no default test coverage. Changing one word can shift behavior globally. Prompts need the discipline we already apply to source: version control, review, eval gates.

**New security surface.** Prompt injection has no reliable defense — only mitigation. A RAG system ingests untrusted retrieved text directly into the prompt, so a document can attempt to carry instructions. The architectural response is privilege separation: retrieved content is data, never authority, and the model's tool access must not depend on what it just read.

**The data flywheel is weaker.** Classical ML logs features plus labels and retrains. For open-ended output you get thumbs up/down — sparse, biased toward angry users, and low-information. Closing the loop is genuinely harder, not just less automated.

**Component coupling.** In RAG, quality is chunking × embedding × index × retrieval × rerank × prompt × model. End-to-end metrics cannot attribute failure, so you need stage-wise instrumentation — which most teams skip, and then fix whichever stage is easiest to measure rather than the one that is broken.

## 2. Where do costs explode?

1. **Agent loops — the biggest one.** Each tool call re-sends the entire growing conversation, so cost is **O(n²) in the number of steps**, not O(n). A 10-step agent over a 20K context burns 200K+ input tokens for a single user request.
2. **Context tokens, not output tokens.** RAG sends 5–20K input tokens to produce 200–500 output. Stuffing the window "to be safe" is the most common and most expensive mistake, and it also *hurts* accuracy via distractors — you pay more to get worse answers.
3. **Prefix cache misses.** If dynamic content (retrieved chunks, a timestamp, a username) precedes the static system prompt, every request pays full prefill. Ordering the prompt static-first is close to free money; a `Date:` line at the top of a system prompt invalidates the cache daily.
4. **Long conversations without compaction.** Per-turn cost grows linearly with turn number, unboundedly.
5. **Retry and self-correction chains.** Every validation retry multiplies the bill.
6. **Evaluation in CI.** A 500-query golden set with LLM-as-judge on every PR is real, recurring spend that nobody budgets.
7. **Re-embedding the corpus.** Changing chunking or embedding model means re-processing 10M documents — and without index versioning you cannot roll back.
8. **Over-provisioned model choice.** A cascade (small model first, escalate on low confidence) typically cuts cost 5–10× at near-equal quality.
9. **Provisioning for peak.** A 10× peak-to-mean ratio means 90% idle GPU if statically provisioned.

Where costs *do not* explode: vector storage and ANN search. Retrieval is under 20% of latency and a negligible fraction of spend — yet that is what teams optimize while the prompt runs to 40K tokens.

## 3. How would you monitor hallucinations and drift?

**Log the full trace first** — query, retrieved chunk IDs with scores, rerank scores, assembled prompt, output, token counts, per-stage latency. Without that trace you cannot attribute a bad answer to retrieval versus generation, and every detector below is guesswork. This is the highest-value operational investment.

**Hallucination — layered, since production has no ground truth:**

- **Groundedness scoring.** Decompose the answer into atomic claims and check each against retrieved context (NLI model or judged LLM). Score = fraction supported. Alert on distribution shift, not on individual values.
- **Citation verification.** Require chunk-ID citations, then *programmatically* verify each citation actually entails the sentence it supports. Cheap, high-signal, and it makes failure auditable rather than vibes.
- **Refusal-rate monitoring.** Confidence gating should produce a steady rate of "not in the documents." A rising rate means content gap or stale corpus; a *falling* rate may mean your threshold stopped protecting you.
- **Self-consistency.** Sample the same query N times at temperature > 0; high variance means the model is guessing. A cheap uncertainty proxy that needs no labels.
- **Judge calibration.** Periodically re-verify the judge against human labels and report agreement. If the judge drifts, all downstream monitoring is fiction.
- **Golden-set regression + canary** on every prompt/model/index change, promoting only if groundedness holds against baseline.

**Drift — four distinct kinds, four detectors, four fixes:**

| Drift | Detector | Fix |
| --- | --- | --- |
| **Query** | PSI/KL on query embeddings vs baseline; new-topic detection via low max-similarity to index | Content expansion, query rewriting, embedding fine-tune |
| **Corpus** *(most common, least monitored)* | Doc-version hashing, last-indexed timestamps, stale-answer sampling | Incremental re-index with TTL and change detection |
| **Retrieval** | Rolling Recall@k on a labeled sample; same query returning different chunks | Index/embedding rollback, chunking review |
| **Model/behavior** | Golden-set deltas; shifts in output length, citation rate, refusal rate | Pin versions, contract tests, eval gates |

Cost and latency are useful *leading* indicators: rising tokens-per-request or TTFT usually precedes the quality problem that causes it.

**What carries over from classical ML:** PSI thresholds, champion/challenger shadow mode, guardrail metrics, threshold re-tuning, and delayed-label proxies all transfer directly — that machinery is sound. **What does not carry over is the label.** In classical ML, drift detection eventually bottoms out in ground truth arriving late. Here it never arrives, so the honest goal is not preventing hallucination but *bounding and detecting* it: constrain generation to retrieved evidence, make every claim traceable to a chunk, and measure the residual rate continuously. Pretending otherwise produces a dashboard that measures fluency and calls it correctness.

**Next action:** wire the trace log and the citation verifier before adding any model capability. Those two make every subsequent failure diagnosable, and without them the eval work in Labs 1–3 stays theoretical.
