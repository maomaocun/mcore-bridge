#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MS_SWIFT_DIR="$(cd "${ROOT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29710}"

if [[ "${SOURCE_MEGATRON_ENV:-1}" != "0" && -f "${MS_SWIFT_DIR}/megatron_env.sh" ]]; then
  # The local ms-swift environment composes Megatron, HF, TE, CUDA wheel paths.
  # Source it when this repo is used from ms-swift/.deps/mcore-bridge.
  # shellcheck source=/dev/null
  source "${MS_SWIFT_DIR}/megatron_env.sh"
fi

export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"

run_check() {
  local offset="$1"
  local script="$2"
  local port=$((MASTER_PORT_BASE + offset))
  echo "[gdn-cp-smoke] ${script} nproc=${NPROC_PER_NODE} master_port=${port}"
  if command -v "${TORCHRUN_BIN}" >/dev/null 2>&1; then
    "${TORCHRUN_BIN}" \
      --nproc_per_node "${NPROC_PER_NODE}" \
      --master_port "${port}" \
      "${ROOT_DIR}/tests/${script}"
  else
    "${PYTHON_BIN}" -m torch.distributed.run \
      --nproc_per_node "${NPROC_PER_NODE}" \
      --master_port "${port}" \
      "${ROOT_DIR}/tests/${script}"
  fi
}

"${PYTHON_BIN}" -m py_compile \
  "${ROOT_DIR}/src/mcore_bridge/model/modules/gated_delta_net.py" \
  "${ROOT_DIR}/tests/check_gdn_cp_a2a.py" \
  "${ROOT_DIR}/tests/check_gdn_cp_backward.py" \
  "${ROOT_DIR}/tests/check_gdn_cp_equivalence.py" \
  "${ROOT_DIR}/tests/check_gdn_cp_fla_equivalence.py"

run_check 0 check_gdn_cp_equivalence.py
run_check 1 check_gdn_cp_backward.py
if [[ "${RUN_FLA:-0}" == "1" ]]; then
  run_check 2 check_gdn_cp_fla_equivalence.py
else
  echo "[gdn-cp-smoke] skip FLA real-operator check; set RUN_FLA=1 to enable"
fi
if [[ "${RUN_SLOW_A2A:-0}" == "1" ]]; then
  run_check 3 check_gdn_cp_a2a.py
else
  echo "[gdn-cp-smoke] skip slow helper-import A2A check; set RUN_SLOW_A2A=1 to enable"
fi

echo "[gdn-cp-smoke] OK"
