#!/bin/bash
# 批量评估所有数据集

# 切换到项目根目录
cd "$(dirname "$0")/.." || exit 1

echo "=========================================="
echo "批量评估所有甲状腺数据集"
echo "=========================================="

# 数据集列表
DATASETS=("TN3K" "DDTI" "ThyroidXL" "PKTN" "TN5K")

# 循环评估每个数据集
for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "=========================================="
    echo "开始评估: ${DATASET}"
    echo "=========================================="

    TRAIN_IMAGE_PATH="./dataset/Segmentation/${DATASET}/train/image"
    TRAIN_MASK_PATH="./dataset/Segmentation/${DATASET}/train/mask"
    TEST_IMAGE_PATH="./dataset/Segmentation/${DATASET}/test/image"
    TEST_MASK_PATH="./dataset/Segmentation/${DATASET}/test/mask"

    CUDA_VISIBLE_DEVICES='0' python main_binary_segmentation.py \
        --train_image_path "${TRAIN_IMAGE_PATH}" \
        --train_mask_path "${TRAIN_MASK_PATH}" \
        --test_image_path "${TEST_IMAGE_PATH}" \
        --test_mask_path "${TEST_MASK_PATH}" \
        --savepath ./output_dir/${DATASET}/ \
        --batch_size 16 \
        --note ${DATASET} \
        --resume ./output_dir/epoch_bestDice.pth \
        --eval

    if [ $? -eq 0 ]; then
        echo "✓ ${DATASET} 评估完成"
    else
        echo "✗ ${DATASET} 评估失败"
    fi
done

echo ""
echo "=========================================="
echo "所有数据集评估完成！"
echo "=========================================="
echo ""
echo "结果保存在 checkpoint 所在目录下自动生成的 eval_<DATASET>_<TIMESTAMP>/ 目录中"
