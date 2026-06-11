# GDN CP Debug Notes

This repo carries experimental context parallel support for Qwen3.x
`gated_delta_net`. Correctness is more important than enabling a larger CP
factor.

## Local Checks

Run the lightweight distributed checks from `.deps/mcore-bridge`:

```bash
source ../../megatron_env.sh  # optional in the local ms-swift/.deps layout
NPROC_PER_NODE=2 tests/run_gdn_cp_smokes.sh
RUN_FLA=1 NPROC_PER_NODE=2 tests/run_gdn_cp_smokes.sh
NPROC_PER_NODE=4 tests/run_gdn_cp_smokes.sh
RUN_SLOW_A2A=1 NPROC_PER_NODE=2 tests/run_gdn_cp_smokes.sh
```

The checks cover:

- CP attention-load-balanced sequence order to hidden-parallel order.
- Packed THD sequence reordering.
- A small GDN-like causal recurrent reference with grouped conv, q/k L2 norm,
  beta/g decay, per-head RMSNorm, output gate, CP roundtrip, and backward
  gradients.
- `RUN_FLA=1` additionally covers the real FLA `causal_conv1d` and
  `chunk_gated_delta_rule` packed path, including backward gradients.

`RUN_SLOW_A2A=1` enables the direct helper-import A2A check. It imports
`gated_delta_net.py` and may spend a long time in FLA/TE import or compile
setup, so the default smoke avoids it.

## 27B Smoke Commands

Use the ms-swift repo root for these commands. Keep `LINEAR_CE_CHUNK_SIZE=0`
for the GDN CP isolation smokes unless the run is specifically validating
chunked linear CE. See `tests/CHUNKED_LINEAR_CE_CP_DEBUG.md` for the separate
CP-safe chunked CE checks and 27B measurements.

```bash
source ./megatron_env.sh

timeout 2400s env \
  SMOKE=1 SFT_CP_LOSS_DEBUG=1 CUDA_VISIBLE_DEVICES=0,1,2,3 \
  NPROC_PER_NODE=4 CONTEXT_PARALLEL_SIZE=1 TENSOR_MODEL_PARALLEL_SIZE=4 \
  PIPELINE_MODEL_PARALLEL_SIZE=1 LINEAR_CE_CHUNK_SIZE=0 \
  GLOBAL_BATCH_SIZE=1 MICRO_BATCH_SIZE=1 MAX_LENGTH=1024 \
  TRUNCATION_STRATEGY=left TRAIN_ITERS=1 NUM_TRAIN_EPOCHS= \
  REPORT_TO=tensorboard WANDB_MODE=disabled SKIP_FINAL_SAVE=true \
  SAVE_STEPS=1000000 RECOMPUTE_GRANULARITY=none RECOMPUTE_METHOD= \
  RECOMPUTE_NUM_LAYERS= SKIP_REASONING_DUP_CHECK=1 \
  DATASET_PATH='data/processed/paper2arm_qwen37_max/paper2arm_qwen37_max_sft_reward_ge_0.6.jsonl#1' \
  RUN_NAME="qwen36-27b-cp1-debug-shapes-$(date +%Y%m%d-%H%M%S)" \
  ./train_qwen36_27b_paper2arm_distill_megatron.sh

timeout 2400s env \
  SMOKE=1 SFT_CP_LOSS_DEBUG=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  NPROC_PER_NODE=8 CONTEXT_PARALLEL_SIZE=2 TENSOR_MODEL_PARALLEL_SIZE=4 \
  PIPELINE_MODEL_PARALLEL_SIZE=1 LINEAR_CE_CHUNK_SIZE=0 \
  GLOBAL_BATCH_SIZE=1 MICRO_BATCH_SIZE=1 MAX_LENGTH=1024 \
  TRUNCATION_STRATEGY=left TRAIN_ITERS=1 NUM_TRAIN_EPOCHS= \
  REPORT_TO=tensorboard WANDB_MODE=disabled SKIP_FINAL_SAVE=true \
  SAVE_STEPS=1000000 RECOMPUTE_GRANULARITY=none RECOMPUTE_METHOD= \
  RECOMPUTE_NUM_LAYERS= SKIP_REASONING_DUP_CHECK=1 \
  DATASET_PATH='data/processed/paper2arm_qwen37_max/paper2arm_qwen37_max_sft_reward_ge_0.6.jsonl#1' \
  RUN_NAME="qwen36-27b-cp2-debug-shapes-$(date +%Y%m%d-%H%M%S)" \
  ./train_qwen36_27b_paper2arm_distill_megatron.sh
```

Useful isolation toggles:

- `PADDING_FREE=false` to bypass the packed THD path.
- `CROSS_ENTROPY_LOSS_FUSION=false` to isolate fused CE reporting.
- `RECOMPUTE_GRANULARITY=full RECOMPUTE_METHOD=uniform RECOMPUTE_NUM_LAYERS=1`
  only for memory pressure checks, not for first correctness checks.

## Interpreting Results

Known validation points from 2026-06-11:

- `check_gdn_cp_a2a.py` passed with CP=2 and CP=4.
- `check_gdn_cp_equivalence.py` passed with CP=2 after RMSNorm was added to the
  reference. CP=4 passed before the RMSNorm reference expansion and should be
  rerun before treating CP=4 as supported.
- After removing the bad merged in-proj early return, `check_gdn_cp_equivalence.py`
  passed with CP=2:
  - unpacked HP output: `max_abs=0.000000`, `max_rel=0.000000`
  - unpacked roundtrip output: `max_abs=0.000000`, `max_rel=0.000000`
  - packed HP output: `max_abs=0.000000`, `max_rel=0.000000`
  - packed roundtrip output: `max_abs=0.000000`, `max_rel=0.000000`
- `check_gdn_cp_backward.py` passed with CP=2:
  - unpacked qkvzba/conv/A_log/dt_bias grads: zero diff at printed precision.
  - unpacked out_norm_weight grad: `max_abs=0.00000001`,
    `max_rel=0.00000045`
  - packed qkvzba/conv/A_log/dt_bias grads: zero diff at printed precision.
  - packed out_norm_weight grad: `max_abs=0.00000001`,
    `max_rel=0.00000031`
- `check_gdn_cp_fla_equivalence.py` passed with CP=2:
  - packed FLA HP output: `max_abs=0.000000`, `max_rel=0.000000`
  - packed FLA roundtrip output: `max_abs=0.000000`, `max_rel=0.000000`
  - packed FLA qkvzba grad: `max_abs=0.000000`, `max_rel=0.007143`
  - packed FLA conv_weight grad: `max_abs=0.000488`, `max_rel=0.005181`
  - packed FLA conv_bias/A_log/dt_bias grads: zero diff at printed precision.
  - packed FLA out_norm_weight grad: `max_abs=0.000488`,
    `max_rel=0.354497`; the relative error is from small denominators, while
    absolute bf16-kernel error remains low.
- A 27B `MAX_LENGTH=1024`, TP=4, CP=2 smoke used the same single sample as the
  CP=1 control: `ignore=400`, `supervised=624`, `first_supervised_index=85`.
- A 27B `MAX_LENGTH=4096`, TP=4, `GLOBAL_BATCH_SIZE=8`,
  `LINEAR_CE_CHUNK_SIZE=0`, full-recompute one-step smoke aligned loss:
  - CP=1: `loss=0.29224876`, `grad_norm=10.20818233`
  - CP=2: `loss=0.29153141`, `grad_norm=10.29060364`
- A TP=8, CP=1, `MAX_LENGTH=4096` one-step smoke after the same fix produced
  `loss=0.29198679`, matching the historical TP=8 smoke `loss=0.29181498`.

## 27B Performance And Capacity

All runs below used BF16, TP=4, PP=1, `GLOBAL_BATCH_SIZE=8`,
`MICRO_BATCH_SIZE=1`, `LINEAR_CE_CHUNK_SIZE=0`, fused CE enabled, padding-free
THD enabled, and selective attention recompute.

| max length | parallelism | GPUs | losses | grad norms | peak GiB/GPU | incremental s/it |
| --- | --- | ---: | --- | --- | ---: | --- |
| 4096 | TP4 CP1 DP1 | 4 | 0.29224876, 0.31905094, 0.27007282, 0.27852163 | 10.21450806, 16.23392677, 8.25052261, 9.17831898 | 42.97 | 109.886, 27.282, 25.424, 24.390 |
| 4096 | TP4 CP2 DP1 | 8 | 0.29193819, 0.31856367, 0.26872766, 0.28082517 | 10.76194477, 15.54413128, 8.92907715, 5.13927412 | 38.46 | 83.040, 15.691, 14.697, 14.741 |
| 4096 | TP4 CP1 DP2 | 8 | 0.29224876, 0.31905094, 0.27162710, 0.27777392 | 10.21164989, 16.23639679, 7.37068462, 5.14653969 | 42.97 | 80.568, 13.353, 13.036, 12.395 |
| 8192 | TP4 CP1 DP2 | 8 | 0.25153366, 0.29436377 | 3.57396030, 4.35069799 | 59.44 | 94.418, 16.307 |
| 8192 | TP4 CP2 DP1 | 8 | 0.25196117, 0.29439181 | 3.91375995, 4.23591614 | 43.43 | 111.261, 18.799 |
| 16384 | TP4 CP2 DP1 | 8 | 0.20072812, 0.22631969 | 1.80706847, 2.15706253 | 59.93 | 116.518, 26.091 |
| 16384 | TP4 CP1 DP2 | 8 | OOM before first step | n/a | 79.16 in use | n/a |

The 16384-token CP1 run failed in TE `linear_fc2` forward while trying to
allocate another 160 MiB; the reported GPU had only about 80 MiB free. The
corresponding CP2 run completed two steps at 59.93 GiB/GPU.

Log roots:

- `logs/qwen36-27b-paper2arm-distill-megatron/qwen36-27b-sft-cp-perf-dp1-4096-20260611-164828-cp1-tp4`
- `logs/qwen36-27b-paper2arm-distill-megatron/qwen36-27b-sft-cp-perf-dp1-4096-20260611-164828-cp2-tp4`
- `logs/qwen36-27b-paper2arm-distill-megatron/qwen36-27b-sft-cp-perf-same8-4096-20260611-170216-cp1-tp4`
- `logs/qwen36-27b-paper2arm-distill-megatron/qwen36-27b-sft-cp-perf-same8-8192-20260611-171517-cp1-tp4`
- `logs/qwen36-27b-paper2arm-distill-megatron/qwen36-27b-sft-cp-perf-same8-8192-20260611-173249-cp2-tp4`
- `logs/qwen36-27b-paper2arm-distill-megatron/qwen36-27b-sft-cp-capacity-16384-20260611-174250-cp2-tp4`
- `logs/qwen36-27b-paper2arm-distill-megatron/qwen36-27b-sft-cp-capacity-16384-20260611-175124-cp1-tp4`

## Production Guidance

- Use CP=2 for long context capacity. In the 8-GPU smoke, 16K fits with CP=2
  and fails with CP=1.
- For 4K/8K when memory fits and throughput is the priority, CP=1 with more DP
  is faster on the same 8 GPUs. CP=2 trades throughput for activation memory.
- `LINEAR_CE_CHUNK_SIZE>0` is numerically CP-safe in the local TP2/CP2 check
  and in the 27B CP2 loss smoke, but the current implementation is not a 27B
  production default because it used more memory and was slower than fused CE
  in the 8K/16K measurements. Keep it opt-in and see
  `tests/CHUNKED_LINEAR_CE_CP_DEBUG.md`.
- Keep `ALLOW_EXPERIMENTAL_CP=true` and `USE_MCORE_GDN=true` explicit in launch
  scripts until this path has longer multi-node burn-in.
- Do not treat CP>2 as production-ready from these checks; the current
  production candidate is CP=2.

Do not treat 27B CP as fully validated unless CP=1 and CP>1 losses align on the
same sample within a small numerical tolerance.
