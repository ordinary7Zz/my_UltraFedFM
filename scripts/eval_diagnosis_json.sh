#!/bin/bash

dataset=toy_diagnosis
nb_classes=2
checkpoint=./output_dir/toy_diagnosis/vit_b_ssl_usffm/log_2024-09-28_16:37:08/checkpoint-best_auroc.pth

CUDA_VISIBLE_DEVICES=1 \
python main_diagnosis.py \
        --model vit_base_patch16 \
        --batch_size 32 \
        --nb_classes ${nb_classes} \
        --data_path ./dataset/${dataset} \
        --resume ${checkpoint} \
        --eval \
        --export_auroc_json
