# Lab 1 — Distributed Training Design (7B Model)

## 1. Problem Statement

Train a 7B-parameter transformer. **Baseline hardware assumed: 8× A100 80GB in a single node, NVLink interconnect** (design changes for other hardware noted inline). Workload: all 7B weights trainable (full SFT / continued pretraining), bf16 mixed precision, Adam.

## 2. The Binding Constraint Is Memory, Not Compute

Per-GPU model states for mixed-precision Adam:

| Component | bytes/param | 7B total |
| --- | --- | --- |
| bf16 params (compute copy) | 2 | 14 GB |
| bf16 gradients | 2 | 14 GB |
| fp32 master weights | 4 | 28 GB |
| fp32 Adam momentum | 4 | 28 GB |
| fp32 Adam variance | 4 | 28 GB |
| **Model states** | **16** | **112 GB** |

Plus activations: ~10–20 GB at micro-batch 2 / seq 2048, falling to ~2–4 GB with gradient checkpointing.

**112 GB > 80 GB.** Naive data parallelism replicates the *full* model on every GPU, so it cannot train 7B on 80GB cards at all. This is the central fact of the design: **gradient synchronization and memory sharding are two orthogonal problems**, and Horovod only solves the first.

## 3. How Many GPUs?

- **Hard floor: 2.** With perfect sharding (ZeRO-3) 112 GB ÷ 80 GB = 2 GPUs, leaving no headroom for activations or a useful batch.
- **Chosen: 8** — one full NVLink node. This is a real cliff, not a round number: intra-node NVLink is ~600 GB/s, while crossing to a second node drops to ~25 GB/s over 200Gb InfiniBand (~24× less). Staying inside one node keeps communication nearly free (§5).
- **Ceiling from efficiency:** at global batches beyond ~1M tokens/step you approach the critical batch size — adding GPUs yields sublinear speedup and forces aggressive LR retuning. 8 keeps us well below that.
- **Scale-out path:** 16–32 GPUs (2–4 nodes) only if the token budget demands it, and only with hierarchical allreduce (§5.4).

## 4. How Is Data Sharded?

Horovod shards **data, never the model**. Rank *r* of *P* reads a disjoint slice via a distributed sampler (`num_replicas=P, rank=r`), or via pre-sharded tar/WebDataset files assigned one-per-rank.

Four invariants must hold or training silently corrupts:

1. **Disjoint** — no sample trained twice in an epoch.
2. **Exhaustive** — every sample seen exactly once.
3. **Consistent shuffle** — all ranks use the same epoch seed and differ only by rank offset; sticky shuffling keeps the rank↔sample mapping stable across restarts.
4. **Equal step counts** — allreduce is a barrier, so a rank with one extra batch **hangs the entire job**. Use `drop_last` or pad with masked samples.

Effective batch = `micro_batch × grad_accum × P`. Changing P changes the batch size, so LR must follow the **linear scaling rule** (LR ∝ global batch) with warmup — large-batch training diverges without it.

Checkpointing: under pure data parallelism all ranks hold identical weights, so **rank 0 alone writes**. Under ZeRO every rank writes its own shard. Persist dataloader state (epoch + sampler position) too, or a resume silently replays data.

## 5. How Does Horovod Synchronize Gradients?

Horovod wraps the base optimizer (`hvd.DistributedOptimizer`): on `step()` it averages every gradient across ranks, then hands the result to Adam. Five mechanisms make that cheap:

**5.1 Identical start.** `hvd.broadcast_parameters(..., root_rank=0)` and `broadcast_optimizer_state` copy rank 0's weights to all ranks at init. Skip this and ranks diverge from step 0 while allreduce keeps forcing their *gradients* to agree — a silent, unrecoverable corruption.

**5.2 Fusion / bucketing.** A 7B model has hundreds of gradient tensors. Sending each separately means per-message latency dominates. Horovod packs gradients into contiguous buffers up to `HOROVOD_FUSION_THRESHOLD` (default 64 MB), paced by `HOROVOD_CYCLE_TIME` (default 5 ms).

**5.3 Ring-allreduce.** Ranks form a ring; each buffer is split into P chunks. Phase 1 **reduce-scatter** (P−1 steps) leaves every rank holding one fully-reduced chunk; phase 2 **allgather** (P−1 steps) distributes them. Per-GPU traffic:

```
2 × M × (P−1)/P        M = gradient payload
```

The `(P−1)/P` term is **independent of P** — adding GPUs does not increase per-GPU bytes. This is why Horovod uses rings instead of a parameter server, where the server would absorb O(P) load and become the bottleneck.

**5.4 Compute/communication overlap.** Autograd produces gradients in reverse layer order, so a bucket's allreduce launches the moment that bucket is ready — *during* backward, not after. For multi-node, hierarchical allreduce reduces within the node over NVLink first, exchanges only across node representatives over IB, then broadcasts back, cutting cross-node traffic ~8×.

**5.5 Averaging and compression.** `op=hvd.Average` divides by world size (if you sum instead, divide LR by P). `Compression.fp16` halves wire bytes at the cost of precision on small gradient magnitudes — pair with error-feedback accumulation if used. For gradient accumulation, `backward_passes_per_step=N` suppresses allreduce until the Nth micro-step, so you pay communication **once per optimizer step**, not N times.

**Is it fast enough?** bf16 gradients = 14 GB/step. Per-GPU ring traffic = `2 × 14 × 7/8` ≈ **24.5 GB**. At 600 GB/s NVLink that is ~0.04 s, overlapped with a ~1.4 s backward pass → **~3% overhead**. Same job across nodes at 25 GB/s → ~1.0 s, which would dominate. This confirms the single-node choice in §3.

Throughput at micro-batch 2 × seq 2048 × 8 GPUs (32k tokens/step), using `6 × N × D` FLOPs:

| Hardware | Effective TFLOPS (40% MFU) | Step time | 10B-token run |
| --- | --- | --- | --- |
| 8× A100 80GB | 125 | ~1.38 s | ~4.9 days |
| 8× H100 80GB | 445 | ~0.39 s | ~1.4 days |

*(Derived from stated MFU and token budget, not a commitment — MFU is the swing factor.)*

## 6. Chosen Design and Framework Contrast

**8× A100 80GB, single node. ZeRO-2 for memory, Horovod/NCCL ring-allreduce for gradient sync, bf16 + gradient checkpointing, micro-batch 2 × seq 2048, grad-accum 4 → global batch 64 seqs (131k tokens), LR linearly scaled with warmup.** DeepSpeed can use Horovod as its communication backend, so the two compose rather than compete.

| Approach | Model states / GPU | Communication | Fits 7B on 8×80GB? |
| --- | --- | --- | --- |
| Horovod (pure DP) | 112 GB — full replica | ring-allreduce of grads | **No** |
| PyTorch DDP | 112 GB | bucketed allreduce via autograd hooks | **No** |
| ZeRO-1 | 38.5 GB | allreduce grads | Yes |
| **ZeRO-2 (chosen)** | **26 GB** | reduce-scatter grads | **Yes** |
| ZeRO-3 / FSDP | 14 GB | allgather params + reduce-scatter grads | Yes, more messages |
| Tensor / pipeline parallel | varies | per-layer P2P | Only when sharding alone is insufficient |

ZeRO-2 over ZeRO-3 because it shards optimizer states *and* gradients while keeping parameters **replicated** — no per-layer allgather on the forward pass, so compute/communication overlap stays clean. ZeRO-3 is the fallback if activations grow (longer sequences).

Horovod vs PyTorch DDP: functionally the same ring-allreduce model. DDP is native and autograd-hook driven; Horovod is framework-agnostic (MPI, TensorFlow/PyTorch/JAX) and adds **Elastic** for spot-preemption recovery — worth it on preemptible capacity, unnecessary on reserved nodes.

## 7. Failure Modes

| Risk | Why it bites | Mitigation |
| --- | --- | --- |
| Straggler rank | Ring allreduce is synchronous — one slow loader stalls all 8 | Per-rank step-time monitoring, prefetch, balanced shard sizes |
| Single node loss | Synchronous job dies entirely | Horovod Elastic host discovery + state broadcast from a survivor |
| NaN on one rank | Allreduce averages it into *every* rank's weights | Per-rank grad-norm logging, skip-step guard, clip |
| Non-reproducible loss | Floating-point reduce is not associative; ring order changes rounding | Pin world size, fusion threshold, and seed |
| Silent data replay | Resume without sampler state | Checkpoint dataloader position; verify on resume |

**Next action:** validate the memory model on one node before any scale-out — run a 200-step smoke test at the target micro-batch and confirm peak per-GPU allocation stays under ~70 GB, then measure achieved MFU against the 40% assumed here and re-derive §5 throughput.
