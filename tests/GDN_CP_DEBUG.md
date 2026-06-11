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
  beta/g decay, per-head RMSNorm, output gate, CP roundtrip.
- `RUN_FLA=1` additionally covers the real FLA `causal_conv1d` and
  `chunk_gated_delta_rule` packed path.

`RUN_SLOW_A2A=1` enables the direct helper-import A2A check. It imports
`gated_delta_net.py` and may spend a long time in FLA/TE import or compile
setup, so the default smoke avoids it.

## 27B Smoke Commands

Use the ms-swift repo root for these commands. Keep `LINEAR_CE_CHUNK_SIZE=0`
while validating CP; chunked linear CE intentionally raises for `CP > 1`.

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
- `check_gdn_cp_equivalence.py` passed with CP=2 and CP=4 before RMSNorm was
  added to the reference; rerun this after edits.
- After removing the bad merged in-proj early return, `check_gdn_cp_equivalence.py`
  passed with CP=2:
  - unpacked HP output: `max_abs=0.000000`, `max_rel=0.000000`
  - unpacked roundtrip output: `max_abs=0.000000`, `max_rel=0.000000`
  - packed HP output: `max_abs=0.000000`, `max_rel=0.000000`
  - packed roundtrip output: `max_abs=0.000000`, `max_rel=0.000000`
- `check_gdn_cp_fla_equivalence.py` passed with CP=2:
  - packed FLA HP output: `max_abs=0.000000`, `max_rel=0.000000`
  - packed FLA roundtrip output: `max_abs=0.000122`, `max_rel=0.007874`
- A 27B `MAX_LENGTH=1024`, TP=4, CP=2 smoke used the same single sample as the
  CP=1 control: `ignore=400`, `supervised=624`, `first_supervised_index=85`.
- A 27B `MAX_LENGTH=4096`, TP=4, `GLOBAL_BATCH_SIZE=8`,
  `LINEAR_CE_CHUNK_SIZE=0`, full-recompute one-step smoke aligned loss:
  - CP=1: `loss=0.29224876`, `grad_norm=10.20818233`
  - CP=2: `loss=0.29153141`, `grad_norm=10.29060364`
- A TP=8, CP=1, `MAX_LENGTH=4096` one-step smoke after the same fix produced
  `loss=0.29198679`, matching the historical TP=8 smoke `loss=0.29181498`.

Do not treat 27B CP as fully validated unless CP=1 and CP>1 losses align on the
same sample within a small numerical tolerance.
