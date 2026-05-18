#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/zjj/code/continual_wsi}"
PYTHON="${PYTHON:-/home/zjj/miniconda3/envs/clam/bin/python}"
RUN_NAME="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${OUT_ROOT:-/data_2_4T/data_zjj/continual_wsi/shortcut_sweeps/${RUN_NAME}}"
LOG_DIR="${OUT_ROOT}/logs"

mkdir -p "${LOG_DIR}"
cd "${PROJECT_DIR}"

CMD=(
  "${PYTHON}" scripts/run_shortcut_sweep.py
  --out-root "${OUT_ROOT}"
  --seeds "${SEEDS:-7,11,13,17,19}"
  --strengths "${STRENGTHS:-2,4,6,8}"
  --l2-lambdas "${L2_LAMBDAS:-20,80}"
  --shortcut-penalties "${SHORTCUT_PENALTIES:-0.0,0.1}"
  --max-workers "${MAX_WORKERS:-3}"
)

printf '%q ' "${CMD[@]}" > "${OUT_ROOT}/launch_command.txt"
printf '\n' >> "${OUT_ROOT}/launch_command.txt"

nohup "${CMD[@]}" > "${LOG_DIR}/sweep.log" 2>&1 &
PID="$!"
echo "${PID}" > "${OUT_ROOT}/sweep.pid"

cat <<EOF
Launched shortcut sweep
PID: ${PID}
Output: ${OUT_ROOT}
Log: ${LOG_DIR}/sweep.log
EOF

