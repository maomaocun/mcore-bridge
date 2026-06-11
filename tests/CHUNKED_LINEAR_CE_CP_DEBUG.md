# Chunked Linear CE CP Debug Notes

This records the context-parallel validation for supervised-token chunked
linear cross entropy. The implementation is CP-safe because the trainer already
slices `hidden_states` and `labels` to the same CP-local attention-load-balanced
sequence order before the model loss path. The CE path should therefore run on
CP-local tokens and reduce only across TP.

Three implementations exist behind `LINEAR_CE_IMPL` when
`LINEAR_CE_CHUNK_SIZE>0`:

- `torch`: the original Python-autograd chunked CE. It materializes a
  per-chunk FP32 logits/exp/grad-logits working set.
- `triton`: a vocab-parallel fused-linear CE chunk path. It still uses a
  per-chunk TP-local logits buffer for the lm_head matmul, but the Triton CE
  kernels compute loss statistics and overwrite that buffer with grad-logits
  in-place. Full sequence logits are not materialized. This is not a fully
  tile-resident zero-logits CUTLASS/TE kernel. The optimized path computes
  per-row max with Triton, avoiding a per-chunk FP32 logits copy, and accumulates
  lm_head weight grad with in-place `addmm_`, avoiding a full-shard temporary
  wgrad tensor per chunk.
- `streaming`: an experimental sequence-parallel aware path. It avoids the
  Megatron output-layer sequence-parallel full hidden all-gather by broadcasting
  each TP owner's supervised-token hidden chunk through the TP group. It still
  materializes per-chunk TP-local logits, so it is not a zero-logits kernel. It
  uses the same row-max and in-place wgrad reductions as `triton`.

## Local Check

Run from `.deps/mcore-bridge`:

```bash
NPROC_PER_NODE=4 TP_SIZE=2 tests/run_chunked_linear_ce_smokes.sh
```

The check compares `LINEAR_CE_IMPL=triton` and the streaming SP-sharded path
against full CE with TP=2 and CP=2. It covers:

- CP-local hidden states and labels using the Megatron attention-load-balanced
  split order.
- TP vocab shards and TP softmax reductions.
- Backward with internal TP input-gradient all-reduce.
- Backward with external sequence-parallel input-gradient all-reduce.
- Streaming hidden states sharded across the TP sequence-parallel dimension.
- Weight-gradient reduction across CP ranks for reference comparison.

Validated on 2026-06-11:

| impl | check | max abs | max rel |
| --- | --- | ---: | ---: |
| torch | loss | 2.4e-7 | 8e-8 |
| torch | hidden grad | 0 | 2.17e-6 |
| torch | weight grad | 1e-8 | 7.7e-5 |
| triton | loss | 2.4e-7 | 8e-8 |
| triton | hidden grad | 0 | 3.25e-6 |
| triton | weight grad | 1e-8 | 9.625e-5 |
| streaming | loss | 2.4e-7 | 8e-8 |
| streaming | hidden grad | 0 | 3.25e-6 |
| streaming | weight grad | 1e-8 | 9.625e-5 |

Both input-gradient paths passed. The current smoke script exercises the
Triton and streaming paths; the `torch` numbers above are retained as the
historical baseline from the same validation round.

## 27B CP2 Smoke Results

All runs used BF16, TP=4, CP=2, PP=1, DP=1, 8 GPUs,
`GLOBAL_BATCH_SIZE=8`, `MICRO_BATCH_SIZE=1`, padding-free THD, selective
attention recompute, and the same 27B SFT smoke script.

| max length | impl | LINEAR_CE_CHUNK_SIZE | losses | grad norms | peak GiB/GPU | incremental s/it |
| --- | --- | ---: | --- | --- | ---: | --- |
| 8192 | none | 0 | 0.25196117, 0.29439181 | 3.91375995, 4.23591614 | 43.43 | 111.261, 18.799 |
| 8192 | torch | 2048 | 0.25196114, 0.29439172 | 3.90876889, 4.22761917 | 47.71 | 86.932, 18.875 |
| 8192 | triton | 2048 | 0.25196111, 0.29439172 | 3.91360569, 4.22973156 | 44.31 | 114.982, 25.760 |
| 8192 | streaming | 2048 | 0.25196111, 0.29439172 | 3.90982723, 4.23182869 | 44.48 | 93.780, 19.653 |
| 16384 | none | 0 | 0.20072812, 0.22631969 | 1.80706847, 2.15706253 | 59.93 | 116.518, 26.091 |
| 16384 | torch | 2048 | 0.20072806, 0.22631963 | 1.80472362, 2.15484047 | 63.90 | 104.408, 28.295 |
| 16384 | torch | 512 | 0.20072731, 0.22631963 | 1.80431390, 2.15794945 | 61.98 | 108.309, 28.862 |
| 16384 | triton | 2048 | 0.20072804, 0.22631963 | 1.80468631, 2.15631199 | 60.36 | 106.952, 27.859 |
| 16384 | triton | 512 | 0.20072731, 0.22631963 | 1.80364609, 2.15686083 | 60.24 | 106.008, 27.077 |
| 16384 | streaming | 2048 | 0.20072804, 0.22631963 | 1.80453157, 2.15973759 | 60.42 | 100.045, 27.257 |
| 16384 | streaming optimized | 2048 | 0.20072804, 0.22631963 | 1.80409408, 2.15695667 | 59.33 | 110.650, 28.331 |

The sample-level supervised-token ratios in these runs were high enough that
supervised-token-only lm_head work has limited room to help:

| max length | supervised tokens | total tokens | ratio |
| --- | ---: | ---: | ---: |
| 8192 | 4641 | 8192 | 56.7% |
| 16384 | 12542 | 16384 | 76.5% |

Log roots:

- `logs/qwen36-27b-paper2arm-distill-megatron/qwen36-27b-sft-cp-perf-same8-8192-20260611-173249-cp2-tp4`
- `logs/qwen36-27b-paper2arm-distill-megatron/qwen36-27b-sft-cp-chunkce-8192-20260611-181043-cp2-tp4`
- `logs/qwen36-27b-paper2arm-distill-megatron/qwen36-27b-sft-cp-capacity-16384-20260611-174250-cp2-tp4`
- `logs/qwen36-27b-paper2arm-distill-megatron/qwen36-27b-sft-cp-chunkce-16384-20260611-181643-cp2-tp4`
- `logs/qwen36-27b-paper2arm-distill-megatron/qwen36-27b-sft-cp-chunkce512-16384-20260611-182340-cp2-tp4`
- `logs/qwen36-27b-paper2arm-distill-megatron/qwen36-27b-sft-cp-fusedlinearce-8192-20260611-192316-cp2-tp4`
- `logs/qwen36-27b-paper2arm-distill-megatron/qwen36-27b-sft-cp-fusedlinearce-16384-20260611-193041-cp2-tp4`
- `logs/qwen36-27b-paper2arm-distill-megatron/qwen36-27b-sft-cp-fusedlinearce512-16384-20260611-193820-cp2-tp4`
- `logs/qwen36-27b-paper2arm-distill-megatron/qwen36-27b-sft-cp-streamingce-8192-20260611-201426-cp2-tp4`
- `logs/qwen36-27b-paper2arm-distill-megatron/qwen36-27b-sft-cp-streamingce-16384-20260611-202400-cp2-tp4`
- `logs/qwen36-27b-paper2arm-distill-megatron/qwen36-27b-sft-cp-streamingce-opt-16384-20260611-2129-cp2-tp4`

## FLA Reference Check

`../flash-linear-attention/fla/modules/fused_linear_cross_entropy.py` is adapted
from Liger Kernel. Its forward loop computes `F.linear(c_x, weight, bias)` per
chunk, then uses Triton CE kernels to compute loss and overwrite chunk logits
with dlogits before computing `dx` and `dw`. This avoids full-sequence logits,
but it still materializes per-chunk logits and is not Megatron vocab-TP safe as
is. It does not provide the fully tile-resident lm_head-to-loss kernel needed to
remove logits entirely.

## Loss-Path Memory Probe

`tests/profile_linear_ce_memory.py` isolates the lm_head plus CE path from the
full 27B model. It initializes a TP-only distributed run and compares:

- `native_mcore`: Megatron sequence-parallel output linear, including its global
  all-gather buffer, followed by native fused vocab-parallel CE.
- `streaming`: `VocabParallelStreamingFusedLinearCrossEntropy`.

The probe reports both `torch.cuda.max_memory_allocated()` and
`torch.cuda.max_memory_reserved()`. Full training logs use
`max_memory_reserved`, so small differences in the table above can include CUDA
caching allocator reservation and fragmentation rather than live tensor usage.

Run examples from `.deps/mcore-bridge`:

```bash
source ../../megatron_env.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 CUDA_DEVICE_MAX_CONNECTIONS=1 \
  LINEAR_CE_PROFILE_CASE=native_mcore PROFILE_SEQ_LEN=16384 PROFILE_IGNORE_TOKENS=3842 \
  torchrun --nproc_per_node 4 --master_port 29785 tests/profile_linear_ce_memory.py

CUDA_VISIBLE_DEVICES=0,1,2,3 CUDA_DEVICE_MAX_CONNECTIONS=1 \
  LINEAR_CE_PROFILE_CASE=streaming PROFILE_SEQ_LEN=16384 PROFILE_IGNORE_TOKENS=3842 \
  PROFILE_CHUNK_SIZE=2048 \
  torchrun --nproc_per_node 4 --master_port 29786 tests/profile_linear_ce_memory.py
```

Validated on 2026-06-11 with TP=4, hidden size 5120, vocab 248320, BF16:

| seq len | case | max allocated GiB | max reserved GiB |
| ---: | --- | ---: | ---: |
| 8192 | native_mcore | 4.4869 | 4.5137 |
| 8192 | streaming optimized | 1.7519 | 1.9941 |
| 16384 | native_mcore | 8.3739 | 8.4004 |
| 16384 | streaming optimized | 1.7911 | 1.8574 |

Before the row-max and in-place wgrad optimization, streaming measured 2.8176 /
3.5371 GiB at 8K and 2.8568 / 3.5371 GiB at 16K. The optimized path removes the
per-chunk FP32 logits copy from `logits.float().amax()` and the full-shard
`grad_weight_chunk` temporary in backward.

This confirms that streaming reduces isolated lm_head+CE live memory. The
optimized full 27B 16K CP2 smoke also reduced end-to-end training
`max_memory_reserved` from the native fused CE baseline 59.93 GiB to 59.33 GiB.
It is therefore no longer only a local loss-path win, but the margin is still
small relative to the total model step peak.

## Production Guidance

- Correctness: CP=2 chunked linear CE aligns with the non-chunked fused CE loss
  to about 1e-6 in the 27B smoke and both implementations pass the TP2/CP2
  gradient check.
- Performance: `LINEAR_CE_IMPL=streaming LINEAR_CE_CHUNK_SIZE=2048` is now a
  production-candidate opt-in for 27B CP2 16K SFT when loss-path memory matters.
  It reduced the measured 16K full-step peak by 0.60 GiB versus native fused CE
  in the local 8-GPU smoke, with matching loss and grad norm. It is still slower
  than native fused CE and needs a 32-GPU staging canary before becoming a
  production default.
- Keep production default `LINEAR_CE_CHUNK_SIZE=0` until the same memory win is
  reproduced on the 32-GPU production topology and representative sequence mix.
- A real production optimization should be a fused lm_head plus CE path that
  avoids materializing even per-chunk logits and avoids the current
  sequence-parallel gather/cublas workspace overhead. That likely belongs in a
  TE/CUTLASS-level kernel, not in this Python-autograd wrapper.
