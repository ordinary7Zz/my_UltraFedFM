CUDA_VISIBLE_DEVICES=0 python main_diagnosis.py \
    --model vit_base_patch16 \
    --batch_size 32 \
    --epochs 10 \
    --nb_classes 2 \
    --data_path /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Lymph_Node_Metastasis_fake/by_LNM_CN01_train \
    --finetune ./output_dir/checkpoint.pth \
    --note LNMCN01_train