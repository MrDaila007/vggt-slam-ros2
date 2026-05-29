#!/usr/bin/env bash
# Batch TUM RGB-D evaluation — Stage 3.4
#
# Runs test_on_tum.py on each sequence found in DATASET_ROOT.
# Outputs per-sequence metrics.txt and a final summary CSV.
#
# Usage:
#   ./scripts/eval_all_tum.sh [DATASET_ROOT] [RESULTS_DIR] [MAX_FRAMES]
#
# Defaults:
#   DATASET_ROOT = data/
#   RESULTS_DIR  = results/
#   MAX_FRAMES   = 0  (all frames)
#
# Example:
#   ./scripts/eval_all_tum.sh data/ results/ 0
#
# Inside Docker (recommended — GPU inference):
#   docker run --rm --runtime nvidia \
#     -e NVIDIA_VISIBLE_DEVICES=all \
#     -v hf_cache:/root/.cache/huggingface \
#     -v $(pwd)/data:/ros2_ws/data:ro \
#     -v $(pwd)/results:/ros2_ws/results:rw \
#     -v $(pwd):/ros2_ws/src/vggt_slam_ros2:ro \
#     -e PYTHONPATH=/opt/vggt:/ros2_ws/src/vggt_slam_ros2 \
#     vggt-slam-ros2:humble \
#     bash /ros2_ws/src/vggt_slam_ros2/scripts/eval_all_tum.sh \
#       /ros2_ws/data/ /ros2_ws/results/ 0

set -euo pipefail

DATASET_ROOT="${1:-data/}"
RESULTS_DIR="${2:-results/}"
MAX_FRAMES="${3:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/test_on_tum.py"

mkdir -p "${RESULTS_DIR}"

SUMMARY_CSV="${RESULTS_DIR}/summary.csv"
echo "sequence,ate_rmse_m,rpe_rmse_m,sim3_scale,matched_poses" > "${SUMMARY_CSV}"

echo "========================================="
echo "VGGT SLAM — TUM RGB-D batch evaluation"
echo "Dataset root : ${DATASET_ROOT}"
echo "Results dir  : ${RESULTS_DIR}"
echo "Max frames   : ${MAX_FRAMES}"
echo "========================================="
echo ""

PASS=0
FAIL=0

for SEQ_DIR in "${DATASET_ROOT}"/rgbd_dataset_freiburg*; do
    [ -d "${SEQ_DIR}" ] || continue
    SEQ_NAME="$(basename "${SEQ_DIR}")"

    echo "--- ${SEQ_NAME} ---"

    if python3 "${EVAL_SCRIPT}" \
            --dataset "${SEQ_DIR}" \
            --out_dir "${RESULTS_DIR}" \
            --max_frames "${MAX_FRAMES}" \
            --no_plot; then

        METRICS_FILE="${RESULTS_DIR}/${SEQ_NAME}_metrics.txt"
        if [ -f "${METRICS_FILE}" ]; then
            ATE=$(grep "RMSE" "${METRICS_FILE}" | head -1 | awk '{print $NF}' | tr -d 'm')
            RPE=$(grep "RMSE" "${METRICS_FILE}" | sed -n '2p' | awk '{print $NF}' | tr -d 'm')
            SIM3=$(grep "Sim3 scale" "${METRICS_FILE}" | awk '{print $NF}')
            # Count matched poses from raw file
            RAW="${RESULTS_DIR}/${SEQ_NAME}_estimated_raw.txt"
            if [ -f "${RAW}" ]; then
                MATCHED=$(grep -c "^[^#]" "${RAW}" || echo "0")
            else
                MATCHED="?"
            fi
            echo "${SEQ_NAME},${ATE},${RPE},${SIM3},${MATCHED}" >> "${SUMMARY_CSV}"
        fi
        PASS=$((PASS + 1))
    else
        echo "  FAILED: ${SEQ_NAME}"
        echo "${SEQ_NAME},ERROR,ERROR,ERROR,0" >> "${SUMMARY_CSV}"
        FAIL=$((FAIL + 1))
    fi

    echo ""
done

echo "========================================="
echo "Batch complete: ${PASS} passed, ${FAIL} failed"
echo ""
echo "Summary table:"
echo "-----------------------------------------"
printf "%-45s %10s %10s %10s\n" "Sequence" "ATE RMSE" "RPE RMSE" "Scale"
echo "-----------------------------------------"
# Skip header line
tail -n +2 "${SUMMARY_CSV}" | while IFS=, read -r seq ate rpe scale matched; do
    printf "%-45s %10s %10s %10s\n" "${seq}" "${ate}" "${rpe}" "${scale}"
done
echo "-----------------------------------------"
echo "Full table saved to: ${SUMMARY_CSV}"
