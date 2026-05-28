DATASET='toy_segmentation' # dataset name
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
    --batch_size 32 \
    --note vit_b_ssl_usffm \
    --resume ./output_dir/toy_segmentation/vit_b_ssl_usffm/log_2024-07-31_16:18:46/epoch_bestDice.pth \
    --eval
