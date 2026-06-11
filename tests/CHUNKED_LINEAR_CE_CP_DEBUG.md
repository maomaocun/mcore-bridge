# Chunked Linear CE CP Debug Notes

This records the context-parallel validation for supervised-token chunked
linear cross entropy. The implementation is CP-safe because the trainer already
slices `hidden_states` and `labels` to the same CP-local attention-load-balanced
sequence order before the model loss path. The CE path should therefore run on
CP-local tokens and reduce only across TP.

Two implementations exist behind `LINEAR_CE_IMPL`:

- `torch`: the original Python-autograd chunked CE. It materializes a
  per-chunk FP32 logits/exp/grad-logits working set.
- `triton`: a vocab-parallel fused-linear CE chunk path. It still uses a
  per-chunk TP-local logits buffer for the lm_head matmul, but the Triton CE
  kernels compute loss statistics and overwrite that buffer with grad-logits
  in-place. Full sequence logits are not materialized. This is not a fully
  tile-resident zero-logits CUTLASS/TE kernel.

## Local Check

Run from `.deps/mcore-bridge`:

```bash
NPROC_PER_NODE=4 TP_SIZE=2 tests/run_chunked_linear_ce_smokes.sh
```

The check compares both `LINEAR_CE_IMPL=torch` and `LINEAR_CE_IMPL=triton`
against full CE with TP=2 and CP=2. It covers:

- CP-local hidden states and labels using the Megatron attention-load-balanced
  split order.
- TP vocab shards and TP softmax reductions.
- Backward with internal TP input-gradient all-reduce.
- Backward with external sequence-parallel input-gradient all-reduce.
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

Both input-gradient paths passed.

## 27B CP2 Smoke Results

All runs used BF16, TP=4, CP=2, PP=1, DP=1, 8 GPUs,
`GLOBAL_BATCH_SIZE=8`, `MICRO_BATCH_SIZE=1`, padding-free THD, selective
attention recompute, and the same 27B SFT smoke script.

| max length | impl | LINEAR_CE_CHUNK_SIZE | losses | grad norms | peak GiB/GPU | incremental s/it |
| --- | --- | ---: | --- | --- | ---: | --- |
| 8192 | none | 0 | 0.25196117, 0.29439181 | 3.91375995, 4.23591614 | 43.43 | 111.261, 18.799 |
| 8192 | torch | 2048 | 0.25196114, 0.29439172 | 3.90876889, 4.22761917 | 47.71 | 86.932, 18.875 |
| 8192 | triton | 2048 | 0.25196111, 0.29439172 | 3.91360569, 4.22973156 | 44.31 | 114.982, 25.760 |
| 16384 | none | 0 | 0.20072812, 0.22631969 | 1.80706847, 2.15706253 | 59.93 | 116.518, 26.091 |
| 16384 | torch | 2048 | 0.20072806, 0.22631963 | 1.80472362, 2.15484047 | 63.90 | 104.408, 28.295 |
| 16384 | torch | 512 | 0.20072731, 0.22631963 | 1.80431390, 2.15794945 | 61.98 | 108.309, 28.862 |
| 16384 | triton | 2048 | 0.20072804, 0.22631963 | 1.80468631, 2.15631199 | 60.36 | 106.952, 27.859 |
| 16384 | triton | 512 | 0.20072731, 0.22631963 | 1.80364609, 2.15686083 | 60.24 | 106.008, 27.077 |

Log roots:

- `logs/qwen36-27b-paper2arm-distill-megatron/qwen36-27b-sft-cp-perf-same8-8192-20260611-173249-cp2-tp4`
- `logs/qwen36-27b-paper2arm-distill-megatron/qwen36-27b-sft-cp-chunkce-8192-20260611-181043-cp2-tp4`
- `logs/qwen36-27b-paper2arm-distill-megatron/qwen36-27b-sft-cp-capacity-16384-20260611-174250-cp2-tp4`
- `logs/qwen36-27b-paper2arm-distill-megatron/qwen36-27b-sft-cp-chunkce-16384-20260611-181643-cp2-tp4`
- `logs/qwen36-27b-paper2arm-distill-megatron/qwen36-27b-sft-cp-chunkce512-16384-20260611-182340-cp2-tp4`
- `logs/qwen36-27b-paper2arm-distill-megatron/qwen36-27b-sft-cp-fusedlinearce-8192-20260611-192316-cp2-tp4`
- `logs/qwen36-27b-paper2arm-distill-megatron/qwen36-27b-sft-cp-fusedlinearce-16384-20260611-193041-cp2-tp4`
- `logs/qwen36-27b-paper2arm-distill-megatron/qwen36-27b-sft-cp-fusedlinearce512-16384-20260611-193820-cp2-tp4`

## Production Guidance

- Correctness: CP=2 chunked linear CE aligns with the non-chunked fused CE loss
  to about 1e-6 in the 27B smoke and both implementations pass the TP2/CP2
  gradient check.
- Performance: do not enable either chunked implementation by default for 27B
  CP2 SFT. The Triton fused-linear chunk path fixes most of the old torch
  chunk memory regression, but it still stayed above the native fused CE peak
  memory in the measured 8K/16K runs and slowed the warm 16K iteration.
- Keep production default `LINEAR_CE_CHUNK_SIZE=0` unless a different model or
  sequence mix shows a clear memory win.
- A real production optimization should be a fused lm_head plus CE path that
  avoids materializing even per-chunk logits and avoids the current
  sequence-parallel gather/cublas workspace overhead. That likely belongs in a
  TE/CUTLASS-level kernel, not in this Python-autograd wrapper.
