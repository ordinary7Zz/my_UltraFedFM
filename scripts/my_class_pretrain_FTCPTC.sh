CUDA_VISIBLE_DEVICES=0 python main_diagnosis.py \
    --model vit_base_patch16 \
    --batch_size 32 \
    --epochs 20 \
    --nb_classes 2 \
    --data_path /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped/by_FTCPTC_train \
    --finetune ./output_dir/checkpoint.pth \
    --note FTCPTC_train