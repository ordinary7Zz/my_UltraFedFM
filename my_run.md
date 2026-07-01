# UltraFedFM 当前运行说明

## 1. 概述

本项目包含两大任务：**分类（classification）** 和 **分割（segmentation）**。

---

# 分类

## 2. 分类入口脚本

| 入口 | 用途 |
|---|---|
| `main_diagnosis.py` | 分类训练 / 评估（`--eval`）/ 导出 AUROC JSON（`--eval --export_auroc_json`） |
| `inference_diagnosis_json.py` | 独立推理，导出 AUROC JSON |
| `inference_flat.py` | 无标签扁平目录推理，输出 CSV |

当前实际在用的 Shell 脚本：
- `scripts/my_class_pretrain.sh`：训练分类模型
- `scripts/eval_all_classification.sh`：批量评估多个数据集
- `scripts/eval_all_classification_json.sh`：批量评估并导出 AUROC JSON

---

## 3. 分类数据集目录结构

使用 `ImageFolder` 结构：

```text
./dataset/Classification/<DATASET>/
├── train/
│   ├── benign/
│   └── malignant/
└── test/
    ├── benign/
    └── malignant/
```

说明：
- `--nb_classes` 设为 `2`
- `train/` 用于训练，`test/` 用于评估
- 子目录名按字典序分配类别编号：推荐 `benign -> 0`，`malignant -> 1`
- 二分类评估默认把 **class index 1** 当作正类
- `inference_diagnosis_json.py` 支持两种目录形式（有 `test/` 则读 `test/`，否则直接读当前目录）

---

## 4. 分类训练

脚本 `scripts/my_class_pretrain.sh` 的核心命令：

```bash
# 二分类
CUDA_VISIBLE_DEVICES=0 python main_diagnosis.py \
  --model vit_base_patch16 \
  --batch_size 32 \
  --epochs 40 \
  --nb_classes 2 \
  --data_path /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/Superimposed_multitask/dataset_3_cls \
  --finetune ./output_dir/checkpoint.pth \
  --note dataset_3_cls_experiment

# 多分类（如 TIRADS 5 分类）
CUDA_VISIBLE_DEVICES=0 python main_diagnosis.py \
  --model vit_base_patch16 \
  --batch_size 16 \
  --epochs 20 \
  --nb_classes 5 \
  --data_path /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/Cine-Clip/Cine-Clip_by_TIRADS/images \
  --finetune ./output_dir/checkpoint.pth \
  --note Cine-Clip_TIRADS
```

关键参数：
- `--finetune`：预训练权重，用于迁移学习（head 层随机初始化，backbone 从预训练权重加载）
- `--resume`：续训权重（恢复 optimizer 和 epoch 状态，断点续训用）

输出目录：
```text
./output_dir/<note>/log_<timestamp>/
```

重点产物：
- `checkpoint-best_acc.pth`
- `checkpoint-best_auroc.pth`
- `logging.log`
- `runs/`

---

## 5. 分类评估

脚本 `scripts/eval_all_classification.sh`，循环评估 DDTI / TN3K / ThyroidXL / TN5K 等数据集。

核心命令：

```bash
CUDA_VISIBLE_DEVICES='0' python main_diagnosis.py \
  --model vit_base_patch16 \
  --batch_size 16 \
  --nb_classes 2 \
  --data_path ./dataset/Classification/${DATASET} \
  --resume ./output_dir/dataset_3_cls_experiment/log_2026-02-27_18:47:44/checkpoint-best_auroc.pth \
  --eval
```

输出目录：
```text
./output_dir/dataset_3_cls_experiment/log_xxx/eval_<DATASET>_<timestamp>/
```

输出文件：
- `roc.csv`
- `overall_stat.csv`
- `confusion_matrix.jpg`
- `logging.log`

---

## 6. 分类：导出 AUROC JSON

使用 `inference_diagnosis_json.py`，配合 `plot_single_task_auroc.py` 绘图。

```bash
CUDA_VISIBLE_DEVICES=0 python inference_diagnosis_json.py \
  --model vit_base_patch16 \
  --batch_size 16 \
  --nb_classes 2 \
  --data_path ./dataset/Classification/finall_data \
  --resume ./output_dir/dataset_3_cls_experiment/log_2026-02-27_18:47:44/checkpoint-best_auroc.pth
```

可选：`--export_json_name my_results.json` 自定义输出文件名。

导出位置：
```text
./output_dir/dataset_3_cls_experiment/log_xxx/eval_<DATASET>_<timestamp>/auroc_results.json
```

JSON 每条记录包含：`true_label`、`predicted_class`、`confidence`、`prob_class_0`、`prob_class_1`、`image_file`、`image_name` 等。

---

## 7. 分类：无标签扁平目录推理

当图片无标签且在同一目录下，使用 `inference_flat.py` 直接输出预测 CSV。

```bash
CUDA_VISIBLE_DEVICES=0 python inference_flat.py \
  --data_path /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/sample/images \
  --resume ./output_dir/dataset_3_cls_experiment/log_2026-02-27_18:47:44/checkpoint-best_auroc.pth \
  --nb_classes 2 \
  --output_csv ./BM.csv
```

| 参数 | 必填 | 说明 |
|---|---|---|
| `--data_path` | 是 | 图片目录，所有图片直接放在该目录下 |
| `--resume` | 是 | 权重 `.pth` 文件路径 |
| `--nb_classes` | 否(默认2) | 分类数 |
| `--model` | 否 | 模型名，默认 `vit_base_patch16` |
| `--batch_size` | 否 | 批量大小，默认 16 |
| `--output_csv` | 否 | 输出路径，默认生成在 checkpoint 同级目录下 |

输出 CSV 列：

| 列名 | 内容 |
|---|---|
| `image_path` | 图片文件名（不含路径） |
| `predicted_class` | 预测类别编号 |
| `confidence` | 该预测类别的概率值 |

---

# 分割

## 8. 分割入口脚本

| 入口 | 用途 |
|---|---|
| `main_binary_segmentation.py` | 二分类分割：训练（默认）/ 评估（`--eval`）/ 可视化（`--plot`） |
当前实际在用的 Shell 脚本：

| 脚本 | 用途 |
|---|---|
| `scripts/my_binary_segmentation_pretrain.sh` | 二分类分割训练 |
| `scripts/my_eval_binary_segmentation_multi.sh` | 二分类分割批量评估 |
| `scripts/plot_binary_segmentation.sh` | 二分类分割结果可视化 |

---

## 9. 分割数据集目录结构

### 二分类分割

图像和掩码通过 4 个独立路径指定：

```text
dataset/
├── train/
│   ├── image/   (--train_image_path)
│   └── mask/    (--train_mask_path)
└── test/
    ├── image/   (--test_image_path)
    └── mask/    (--test_mask_path)
```

图像和掩码通过文件名 stem 自动匹配，掩码读取为灰度图（像素值 ÷ 255 归一化到 [0,1]）。

---

## 10. 二分类分割

### 10.1 训练

```bash
./scripts/my_train_seg_nodule.sh
./scripts/my_train_seg_gland.sh
```

关键参数：
- `--pretrained`：预训练 MAE checkpoint 路径
- 输出路径：`<savepath>/<note>/log_<timestamp>/`

评估指标：IoU、Dice、MAE、Hausdorff Distance (HD95)，含 Bootstrap 置信区间。

### 10.2 评估

```bash
python main_binary_segmentation.py \
    --train_image_path '/mnt/wangbd8/workspace/DataSets/ThyroidAgent/TGVideo_PNG/train/image' \
    --train_mask_path '/mnt/wangbd8/workspace/DataSets/ThyroidAgent/TGVideo_PNG/train/mask' \
    --test_image_path '/mnt/wangbd8/workspace/DataSets/ThyroidAgent/TGVideo_PNG/test/image' \
    --test_mask_path '/mnt/wangbd8/workspace/DataSets/ThyroidAgent/TGVideo_PNG/test/mask' \
    --savepath ./my_infer_output/seg/gland \
    --note vit_b_ssl_usffm \
    --resume ./my_pth/gland_seg/epoch_bestDice.pth \
    --eval

python main_binary_segmentation.py \
    --train_image_path /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/sample/images \
    --train_mask_path /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/sample/masks \
    --test_image_path /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/sample/images \
    --test_mask_path /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/sample/masks \
    --savepath ./my_infer_output/sample \
    --note vit_b_ssl_usffm \
    --resume ./my_pth/nodule_seg/epoch_bestDice.pth \
    --eval
```

批量评估脚本：`scripts/my_eval_binary_segmentation_multi.sh`，对 Augtrain / finall_data / TN3K / DDTI / ThyroidXL / PKTN / TN5K 共 7 个数据集依次评估。

### 10.3 可视化
TRAIN_IMAGE_PATH='/mnt/wangbd8/workspace/DataSets/ThyroidAgent/TGVideo_PNG/train/image'
TRAIN_MASK_PATH='/mnt/wangbd8/workspace/DataSets/ThyroidAgent/TGVideo_PNG/train/mask'
TEST_IMAGE_PATH='/mnt/wangbd8/workspace/DataSets/ThyroidAgent/TGVideo_PNG/test/image'
TEST_MASK_PATH='/mnt/wangbd8/workspace/DataSets/ThyroidAgent/TGVideo_PNG/test/mask'
```bash
python main_binary_segmentation.py \
    --train_image_path ./dataset/xxx/train/image \
    --train_mask_path ./dataset/xxx/train/mask \
    --test_image_path ./dataset/xxx/test/image \
    --test_mask_path ./dataset/xxx/test/mask \
    --savepath ./output_dir/xxx \
    --note vit_b_ssl_usffm \
    --resume ./output_dir/xxx/.../epoch_bestDice.pth \
    --plot
```

### 10.4 无标签扁平目录推理

当只有一批无标签图片，需要直接输出预测掩码时，使用 `inference_seg_flat.py`：

```bash
CUDA_VISIBLE_DEVICES=0 python inference_seg_flat.py \
  --data_path /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/sample/images \
  --resume ./my_pth/nodule_seg/epoch_bestDice.pth \
  --output_dir ./my_infer_output/nodule_masks
```

| 参数 | 必填 | 说明 |
|---|---|---|
| `--data_path` | 是 | 图片目录，所有图片在同一目录下 |
| `--resume` | 是 | 分割模型 `.pth` 权重路径 |
| `--output_dir` | 是 | 掩码输出目录 |
| `--img_size` | 否 | 输入尺寸，默认 224 |
| `--batch_size` | 否 | 默认 1 |
| `--threshold` | 否 | 二值化阈值，默认 0.5 |

掩码保存为 PNG，文件名与原图相同（扩展名改为 `.png`），尺寸与原图一致。

---

# 总结

## 11. 推荐使用顺序

1. 准备数据集（按对应目录结构）
2. 如有预训练 MAE 权重，先放到对应路径
3. 运行训练脚本
4. 用评估脚本做批量评估
5. 分类可额外导出 AUROC JSON 或无标签推理 CSV
6. 分割可运行可视化脚本或 `inference_seg_flat.py` 输出掩码

## 12. 一句话总结

| 任务 | 训练 | 评估 | 额外能力 |
|---|---|---|---|
| 分类 | `main_diagnosis.py` | `main_diagnosis.py --eval` | 导出 AUROC JSON / 无标签推理 CSV |
| 二分类分割 | `main_binary_segmentation.py` | `main_binary_segmentation.py --eval` | `--plot` 可视化 / 无标签推理导出掩码 |
