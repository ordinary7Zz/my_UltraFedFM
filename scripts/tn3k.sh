cd "$(dirname "$0")/.." || exit 1

echo "结果将保存在 checkpoint 所在目录下自动生成的 eval_TN3K_<TIMESTAMP>/ 目录中"

DATASET='TN3K' # dataset name
TRAIN_IMAGE_PATH="./dataset/${DATASET}/train/image"
TRAIN_MASK_PATH="./dataset/${DATASET}/train/mask"
TEST_IMAGE_PATH="./dataset/${DATASET}/test/image"
TEST_MASK_PATH="./dataset/${DATASET}/test/mask"
CUDA_VISIBLE_DEVICES='0' python main_binary_segmentation.py \
    --train_image_path "${TRAIN_IMAGE_PATH}" \
    --train_mask_path "${TRAIN_MASK_PATH}" \
    --test_image_path "${TEST_IMAGE_PATH}" \
    --test_mask_path "${TEST_MASK_PATH}" \
    --savepath ./output_dir/${DATASET} \
    --batch_size 16 \
    --note ${DATASET} \
    --resume ./output_dir/epoch_bestDice.pth \
    --eval
