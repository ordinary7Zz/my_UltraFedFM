#!/usr/bin/env bash
set -euo pipefail

# Run from project root
cd "$(dirname "$0")/.."

# ====== Config ======
DATASETS=("Augtrain" "finall_data" "TN3K" "DDTI" "ThyroidXL" "PKTN" "TN5K")
DATA_ROOT="./dataset/Segmentation"
CUDA_DEVICE="0"
BATCH_SIZE=16
NOTE="vit_b_ssl_usffm"

# Change this to your trained pth
RESUME_PATH="./output_dir/dataset_4_seg/vit_b_ssl_usffm/log_2026-05-31_07:35:05/epoch_bestDice.pth"

# Optional: where to store console logs
LOG_DIR="./output_dir/eval_logs"
mkdir -p "${LOG_DIR}"

if [[ ! -f "${RESUME_PATH}" ]]; then
    echo "Checkpoint not found: ${RESUME_PATH}"
    exit 1
fi

echo "=========================================="
echo "Batch evaluation for binary segmentation"
echo "=========================================="
echo "Resume: ${RESUME_PATH}"
echo "Datasets: ${DATASETS[*]}"
echo ""

for DATASET in "${DATASETS[@]}"; do
    echo "=========================================="
    echo "Evaluating: ${DATASET}"
    echo "=========================================="

    DATA_DIR="${DATA_ROOT}/${DATASET}"
    TRAIN_IMAGE_PATH="${DATA_DIR}/train/image"
    TRAIN_MASK_PATH="${DATA_DIR}/train/mask"
    TEST_IMAGE_PATH="${DATA_DIR}/test/image"
    TEST_MASK_PATH="${DATA_DIR}/test/mask"
    SAVE_PATH="./output_dir/${DATASET}/${NOTE}"
    RUN_LOG="${LOG_DIR}/eval_${DATASET}.log"

    mkdir -p "${SAVE_PATH}"

    CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" python main_binary_segmentation.py \
        --train_image_path "${TRAIN_IMAGE_PATH}" \
        --train_mask_path "${TRAIN_MASK_PATH}" \
        --test_image_path "${TEST_IMAGE_PATH}" \
        --test_mask_path "${TEST_MASK_PATH}" \
        --savepath "${SAVE_PATH}" \
        --batch_size "${BATCH_SIZE}" \
        --note "${NOTE}" \
        --resume "${RESUME_PATH}" \
        --eval 2>&1 | tee "${RUN_LOG}"

    echo "Done: ${DATASET}"
    echo "Log saved to: ${RUN_LOG}"
    echo ""
done

echo "=========================================="
echo "All datasets finished"
echo "=========================================="
