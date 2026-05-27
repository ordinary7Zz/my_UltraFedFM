CUDA_VISIBLE_DEVICES=0 python main_diagnosis.py \
    --model vit_base_patch16 \
    --batch_size 32 \
    --epochs 40 \
    --nb_classes 2 \
    --data_path /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/Superimposed_multitask/dataset_3_cls \
    --finetune ./output_dir/checkpoint.pth \
    --note dataset_3_cls_experiment