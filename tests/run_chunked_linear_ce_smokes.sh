#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MS_SWIFT_DIR="$(cd "${ROOT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-29730}"
TP_SIZE="${TP_SIZE:-2}"

if [[ "${SOURCE_MEGATRON_ENV:-1}" != "0" && -f "${MS_SWIFT_DIR}/megatron_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${MS_SWIFT_DIR}/megatron_env.sh"
fi

export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
export TP_SIZE

"${PYTHON_BIN}" -m py_compile \
  "${ROOT_DIR}/src/mcore_bridge/model/fused_linear_ce.py" \
  "${ROOT_DIR}/src/mcore_bridge/model/gpt_model.py" \
  "${ROOT_DIR}/tests/check_chunked_linear_ce_cp.py"

echo "[chunked-linear-ce-smoke] check_chunked_linear_ce_cp.py nproc=${NPROC_PER_NODE} tp=${TP_SIZE} master_port=${MASTER_PORT}"
if command -v "${TORCHRUN_BIN}" >/dev/null 2>&1; then
  "${TORCHRUN_BIN}" \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --master_port "${MASTER_PORT}" \
    "${ROOT_DIR}/tests/check_chunked_linear_ce_cp.py"
else
  "${PYTHON_BIN}" -m torch.distributed.run \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --master_port "${MASTER_PORT}" \
    "${ROOT_DIR}/tests/check_chunked_linear_ce_cp.py"
fi

echo "[chunked-linear-ce-smoke] OK"
