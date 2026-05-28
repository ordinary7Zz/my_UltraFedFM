DATASET='dataset_3_seg' # dataset name
TRAIN_IMAGE_PATH='/mnt/wangbd8/workspace/DataSets/ThyroidAgent/Superimposed_experiment/dataset_3/train/image'
TRAIN_MASK_PATH='/mnt/wangbd8/workspace/DataSets/ThyroidAgent/Superimposed_experiment/dataset_3/train/mask'
TEST_IMAGE_PATH='/mnt/wangbd8/workspace/DataSets/ThyroidAgent/Superimposed_experiment/dataset_3/test/image'
TEST_MASK_PATH='/mnt/wangbd8/workspace/DataSets/ThyroidAgent/Superimposed_experiment/dataset_3/test/mask'
CUDA_VISIBLE_DEVICES='1' python main_binary_segmentation.py \
    --train_image_path "${TRAIN_IMAGE_PATH}" \
    --train_mask_path "${TRAIN_MASK_PATH}" \
    --test_image_path "${TEST_IMAGE_PATH}" \
    --test_mask_path "${TEST_MASK_PATH}" \
    --savepath ./output_dir/${DATASET} \
    --batch_size 96 \
    --epoch 10 \
    --note vit_b_ssl_usffm \
    --pretrained ./output_dir/pretrained_ultrafedfm/log_2024-07-16_13:53:08/checkpoint.pth