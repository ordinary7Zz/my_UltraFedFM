DATASET='dataset_3_seg' # dataset name
CUDA_VISIBLE_DEVICES='1' python main_binary_segmentation.py \
    --datapath /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Superimposed_experiment/dataset_3/images \
    --savepath ./output_dir/${DATASET} \
    --batch_size 96 \
    --epoch 10 \
    --note vit_b_ssl_usffm \
    --pretrained ./output_dir/pretrained_ultrafedfm/log_2024-07-16_13:53:08/checkpoint.pth