# Lab 2 — Inference Stack Design

## 1. Framing: These Are Three Different Layers

The comparison is a trap if treated as three points on one axis. They occupy different layers of the stack:

| | What it actually is |
| --- | --- |
| **FastAPI + single GPU** | A **web/transport layer** — routing, auth, request shaping, streaming. Inference is whatever you call in-process. |
| **Triton Inference Server** | A **serving orchestrator** — multi-model repository, versioning, dynamic batching, ensembles, metrics, gRPC/HTTP. Framework-agnostic. |
| **vLLM** | An **inference engine** — KV-cache memory management and iteration-level scheduling for autoregressive decode. LLM-only. |

They compose rather than compete: a production chat service is usually `FastAPI (business logic, auth) → vLLM or Triton+TensorRT-LLM (engine) → GPU`. The genuine decision is **which engine**, and **whether you need an orchestrator above it**.

**Assumed context (carried from Lab 1):** 7B model, bf16, 8× A100 80GB single node. Note a 7B model needs ~14 GB of weights, so it fits on one GPU with ~66 GB free for KV cache — **tensor parallelism is the wrong tool here; run independent replicas and load-balance.** TP splits weights you did not need to split and adds an allreduce per layer.

## 2. Comparison Table

| Dimension | FastAPI + single GPU | Triton Inference Server | vLLM |
| --- | --- | --- | --- |
| **Batching** | None by default; static batching if hand-rolled | Dynamic batching (queue-delay window, fixed shapes); in-flight batching only via TRT-LLM/vLLM backend | **Continuous batching** — requests join/leave the running batch every decode step |
| **KV cache memory** | Preallocated to `max_seq_len` per slot → **60–80% wasted** to fragmentation | Backend-dependent | **PagedAttention**, block-allocated → **<4% waste** |
| **Concurrent sequences (7B on 80GB)** | ~10–25 in practice | Depends on backend | **~60+** (bandwidth-limited, not memory-limited) |
| **Latency at QPS≈1** | **Best** — direct in-process call, no scheduler or IPC hop | Good; gRPC + fused kernels, one extra hop | Near-identical to FastAPI; small scheduler overhead |
| **Latency under concurrency** | **Degrades badly** — requests queue behind a serialized `generate()` | Good with TRT-LLM backend | **Best-in-class for text** — ITL stays flat, chunked prefill protects TTFT |
| **TTFT levers** | None | Backend-dependent | Prefix caching, chunked prefill, priority scheduling |
| **Streaming** | Manual (SSE/WebSocket) | Sequence batching + correlation IDs | Native, OpenAI-compatible |
| **Multi-framework / multi-model** | No — one process, one model | **Yes** — TensorRT, ONNX, PyTorch, TF, Python, vLLM, TRT-LLM in one repository | No — autoregressive LLMs only |
| **Ensembles / pipelines** | Hand-written in Python | **Yes** — DAGs, BLS (embed → rerank → generate) | No |
| **Versioning / A-B / canary** | Hand-rolled | **Yes** — model repository, version pinning, warmup | Limited (external router needed) |
| **Observability** | Hand-rolled | **Yes** — Prometheus, per-model stats, rate limiters | Prometheus metrics, basic |
| **Multi-GPU scaling** | Manual processes + external LB | Concurrent model instances, MIG/MPS multi-tenancy | TP/PP, data-parallel replicas, **multi-LoRA hot-swap** |
| **Quantization** | Manual | TRT-LLM FP8/INT4 (fastest kernels, needs engine build) | AWQ/GPTQ/FP8, KV-cache quantization |
| **Ops complexity** | **Lowest** | **Highest** — `config.pbtxt`, repository layout, C++ debugging | Low — `vllm serve`, one command |
| **Dev velocity** | **Highest** | Low | High |

## 3. The Physics Behind the Decision

Two numbers explain most of the table.

**(a) Decode is memory-bandwidth bound, not compute bound.** Generating one token requires reading all 14 GB of weights. A100 HBM is ~2 TB/s, so the floor is `14 GB / 2 TB/s ≈ 7 ms/token ≈ 143 tok/s` for a single stream — **regardless of how much FLOPs you have**. Batching amortizes that one weight read across the whole batch, which is why concurrency buys throughput almost for free until you saturate bandwidth.

**(b) But batching costs inter-token latency.** Each step also reads the KV cache. For a 7B model (32 layers, 32 KV heads, dim 128, bf16) KV is **0.5 MB/token**, so at 2K context each sequence costs 1 GB:

| Batch | Bytes read / step | ITL | Aggregate |
| --- | --- | --- | --- |
| 1 | 14 GB (weights) | ~7 ms | 143 tok/s |
| 8 | 22 GB | ~11 ms | 727 tok/s |
| 32 | 46 GB | ~23 ms | 1,391 tok/s |
| 64 | 78 GB | ~39 ms | 1,641 tok/s |

This is the whole tradeoff in one table: **batching trades per-user latency for aggregate throughput**, and the exchange rate is set by KV bytes per token. GQA (8 KV heads → 0.125 MB/token) or FP8 KV cache shifts it ~4× in your favor — at batch 64 with GQA, ITL drops to ~15 ms and aggregate rises to ~4,300 tok/s.

**Consequence:** low-latency chat wants a *small* batch and a scheduler that protects ITL; high-throughput batch wants the *largest* batch memory allows. No static-batching server can serve both, because it must commit to a batch shape up front and run every member to the longest length — padding waste plus head-of-line blocking. Continuous batching resolves this by re-forming the batch every iteration.

## 4. Decision — Low-Latency Chat

**vLLM.** Chat is dominated by TTFT and ITL at low-to-moderate concurrency, with streaming output and a shared system prompt across requests.

- **Prefix caching** reuses the KV of the identical system prompt across every conversation — on a long system prompt this removes most of the prefill work and cuts TTFT dramatically. Neither FastAPI nor vanilla Triton does this.
- **Chunked prefill** splits a long prompt into pieces interleaved with decode steps, so one user's 8K-token paste does not freeze everyone else's token stream. This is the single most important latency-under-load feature and it exists specifically for chat.
- **Continuous batching** keeps ITL flat as concurrency rises, where FastAPI's serialized `generate()` degrades linearly.
- Native OpenAI-compatible streaming, so no hand-rolled SSE plumbing.

**When Triton wins instead:** if you need the absolute minimum per-token latency and will pay for it — Triton + **TensorRT-LLM** backend has faster fused kernels than vLLM, and if your chat pipeline is a *DAG* (embed → retrieve → rerank → generate) Triton's ensembles express that natively while vLLM cannot. Also choose Triton if the same server must host CV/ASR/embedding models alongside the LLM.

**FastAPI + single GPU** is defensible for chat only at prototype scale (single-digit QPS, no SLA), where its zero-hop in-process call actually has the lowest latency of the three — and where you value total control over custom post-processing, tool-calling logic, and guardrails more than throughput.

## 5. Decision — High-Throughput Batch Inference

**Split by whether there is a latency SLA.**

- **Offline batch (no SLA) → vLLM's `LLM` Python class directly, no server at all.** Feed the whole dataset in; continuous batching saturates the GPU and you skip HTTP, queueing, and a network hop entirely. This is the highest-throughput configuration available and the one most often missed — people deploy a server in front of a job that never needed one.
- **Online high-concurrency (SLA, but throughput-weighted) → vLLM, data-parallel replicas.** Run 8 independent single-GPU replicas (TP=1) behind a router rather than one TP=8 instance; you get 8× the KV-cache budget and no per-layer allreduce. Push batch size until ITL hits your SLA ceiling (§3 gives the exchange rate).
- **Heterogeneous or multi-tenant batch → Triton.** If the batch workload mixes model types (CV + embeddings + LLM), needs GPU sharing via MIG/MPS with concurrent model instances, or requires per-model versioning and Prometheus accounting across teams, Triton's orchestrator layer earns its complexity. vLLM cannot host a ResNet.

**Never FastAPI for throughput.** Static batching with preallocated KV cache caps you near ~10–25 concurrent sequences and wastes 60–80% of GPU memory on fragmentation. It is the worst option on this axis by a wide margin.

## 6. Recommended Architecture

```
Client → FastAPI gateway (auth, rate limit, tool/guardrail logic, SSE)
              → vLLM engine, TP=1, one replica per GPU (continuous batching,
                prefix caching, chunked prefill)
              → offline jobs: vLLM LLM class, no gateway
```

Add Triton **only** when you acquire a second model type or need ensemble DAGs and formal versioning — at which point mount vLLM/TRT-LLM as its backend rather than replacing it.

## 7. Failure Modes

| Risk | Mitigation |
| --- | --- |
| ITL collapse under load as batch grows | Cap `max_num_seqs` from the §3 table; alert on p95 ITL, not just throughput |
| KV cache OOM at long contexts | Bound `max_model_len`; enable FP8 KV cache; reject rather than queue pathological prompts |
| One long prefill stalling all decodes | Chunked prefill on; separate prefill/decode pools if severe |
| Wasted TP on a 7B model | TP=1 + data-parallel replicas (§1) |
| GPU idle behind a Python event loop | Tokenization/serialization off the critical path; multiple uvicorn workers per replica is wrong — one engine per GPU |
| No backpressure | 429 on queue depth; do not let requests pile into an unbounded queue |

**Next action:** benchmark before committing — replay a captured production trace (real prompt/response length distributions, not synthetic uniform lengths) against vLLM and Triton+TRT-LLM on one A100, and measure the ITL-vs-batch curve empirically. The §3 table is derived from bandwidth arithmetic and will be directionally right, but the crossover point where your SLA breaks is workload-specific and is the number that actually decides this.
