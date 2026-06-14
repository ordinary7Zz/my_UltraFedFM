#!/bin/bash
# 批量评估所有甲状腺分类数据集

# 切换到项目根目录
cd "$(dirname "$0")/.." || exit 1

echo "=========================================="
echo "批量评估甲状腺良恶性分类任务"
echo "=========================================="

# 数据集列表
DATASETS=("finall_data" "DDTI" "TN3K" "ThyroidXL" "TN5K")
NB_CLASSES=2  # 良性/恶性二分类

# 循环评估每个数据集
for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "=========================================="
    echo "开始评估: ${DATASET}"
    echo "=========================================="
    
    CUDA_VISIBLE_DEVICES='0' python main_diagnosis.py \
        --model vit_base_patch16 \
        --batch_size 16 \
        --nb_classes ${NB_CLASSES} \
        --data_path ./dataset/Classification/${DATASET} \
        --resume ./output_dir/dataset_3_cls_experiment/log_2026-02-27_18:47:44/checkpoint-best_auroc.pth \
        --eval
    
    if [ $? -eq 0 ]; then
        echo "✓ ${DATASET} 分类评估完成"
    else
        echo "✗ ${DATASET} 分类评估失败"
    fi
done

echo ""
echo "=========================================="
echo "所有数据集分类评估完成！"
echo "=========================================="
echo ""
echo "结果保存在 checkpoint 所在目录下自动生成的 eval_<DATASET>_<TIMESTAMP>/ 目录中"
