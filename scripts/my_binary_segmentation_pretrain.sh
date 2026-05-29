DATASET='dataset_4_seg' # dataset name
TRAIN_IMAGE_PATH='/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/Superimposed_experiment/dataset_4/train/images'
TRAIN_MASK_PATH='/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/Superimposed_experiment/dataset_4/train/masks'
TEST_IMAGE_PATH='/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/Superimposed_experiment/dataset_4/test/images'
TEST_MASK_PATH='/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/Superimposed_experiment/dataset_4/test/masks'
CUDA_VISIBLE_DEVICES='1' python main_binary_segmentation.py \
    --train_image_path "${TRAIN_IMAGE_PATH}" \
    --train_mask_path "${TRAIN_MASK_PATH}" \
    --test_image_path "${TEST_IMAGE_PATH}" \
    --test_mask_path "${TEST_MASK_PATH}" \
    --savepath ./output_dir/${DATASET} \
    --batch_size 96 \
    --epoch 10 \
    --note vit_b_ssl_usffm \
    --pretrained ./output_dir/epoch_bestDice.pth