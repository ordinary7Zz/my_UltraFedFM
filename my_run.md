# UltraFedFM 当前运行说明

## 1. 这份说明保留什么
这份文件只保留**当前实际在用**的甲状腺良恶性二分类流程。

不再写入以下内容：
- `scripts/` 下默认指向 `toy_diagnosis`、`toy_pretrain`、`toy_segmentation`、`toy_multi_segmentation` 的示例脚本
- 未实际使用过的 toy 数据跑通流程
- 与当前分类任务无关的预训练/分割说明

---

## 2. 当前任务入口
当前主要入口都是分类脚本：

- 训练/微调入口：`main_diagnosis.py`
- 评估入口：`main_diagnosis.py --eval`
- 导出 AUROC JSON 入口：`main_diagnosis.py --eval --export_auroc_json`

当前实际在用的脚本：
- `scripts/my_class_pretrain.sh`：训练当前分类模型
- `scripts/eval_all_classification.sh`：批量评估多个甲状腺分类数据集
- `scripts/eval_all_classification_json.sh`：批量评估多个甲状腺分类数据集，并额外导出 AUROC JSON

---

## 3. 分类数据集目录结构
分类任务使用 `ImageFolder` 目录结构。

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
- `train/` 用于训练
- `test/` 用于评估/推理
- 每个类别目录下直接放图片

---

## 4. 当前训练方式
当前训练脚本：`scripts/my_class_pretrain.sh`

脚本内容对应的实际命令为：

```bash
CUDA_VISIBLE_DEVICES=0 python main_diagnosis.py \
  --model vit_base_patch16 \
  --batch_size 32 \
  --epochs 40 \
  --nb_classes 2 \
  --data_path /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/Superimposed_multitask/dataset_3_cls \
  --finetune ./output_dir/checkpoint.pth \
  --note dataset_3_cls_experiment
```

训练输出目录形如：

```text
./output_dir/dataset_3_cls_experiment/log_<timestamp>/
```

重点产物：
- `checkpoint-best_acc.pth`
- `checkpoint-best_auroc.pth`
- `logging.log`
- `runs/`

---

## 5. 当前批量评估方式
当前批量评估脚本：`scripts/eval_all_classification.sh`

它会循环评估以下数据集：
- `DDTI`
- `TN3K`
- `ThyroidXL`
- `TN5K`

核心命令形式：

```bash
CUDA_VISIBLE_DEVICES='0' python main_diagnosis.py \
  --model vit_base_patch16 \
  --batch_size 16 \
  --nb_classes 2 \
  --data_path ./dataset/Classification/${DATASET} \
  --resume ./output_dir/dataset_3_cls_experiment/log_2026-02-27_18:47:44/checkpoint-best_auroc.pth \
  --eval
```

每次评估会在 checkpoint 同级目录自动生成：

```text
./output_dir/dataset_3_cls_experiment/log_xxx/eval_<DATASET>_<timestamp>/
```

默认输出：
- `roc.csv`
- `overall_stat.csv`
- `confusion_matrix.jpg`
- `logging.log`

---

## 6. 当前新增：导出 AUROC 绘图 JSON
为了给 `plot_single_task_auroc.py` 使用，现在评估时支持额外导出标准 JSON。

### 单数据集导出方式

```bash
CUDA_VISIBLE_DEVICES=0 python main_diagnosis.py \
  --model vit_base_patch16 \
  --batch_size 16 \
  --nb_classes 2 \
  --data_path ./dataset/Classification/DDTI \
  --resume ./output_dir/dataset_3_cls_experiment/log_2026-02-27_18:47:44/checkpoint-best_auroc.pth \
  --eval \
  --export_auroc_json
```

如果想自定义输出文件名，还可以加：

```bash
--export_json_name my_results.json
```

### 批量导出方式

当前新增脚本：`scripts/eval_all_classification_json.sh`

它与 `scripts/eval_all_classification.sh` 的评估参数保持一致，只额外增加：

```bash
--export_auroc_json
```

也就是说：
- 模型相同
- `batch_size` 相同
- `nb_classes` 相同
- `data_path` 相同
- `resume` 相同
- 评估的数据集列表相同
- 只是在每次评估目录下多导出一个 JSON 文件

### 导出位置
仍然保存在评估目录下：

```text
./output_dir/dataset_3_cls_experiment/log_xxx/eval_<DATASET>_<timestamp>/auroc_results.json
```

### 当前 JSON 内容
导出的 JSON 顶层是一个列表，每条记录至少包含：
- `true_label`
- `prob_class_1`

同时还会尽量带上：
- `record_type`
- `selected_model`
- `predicted_class`
- `confidence`
- `prob_class_0`
- `image_file`
- `image_name`

该格式可直接对接 `plot_single_task_auroc.py` 所要求的样本级结果格式。

---

## 7. 推荐使用顺序
如果现在要复现当前流程，建议按下面顺序：

1. 准备二分类目录结构数据集
2. 运行 `scripts/my_class_pretrain.sh` 训练模型
3. 用 `scripts/eval_all_classification.sh` 做批量评估
4. 如果要保留可直接画 AUROC 的结果，运行 `scripts/eval_all_classification_json.sh` 或手动在评估命令后加 `--export_auroc_json`

---

## 8. 一句话总结
当前项目里，甲状腺良恶性二分类的最新用法是：

- 用 `main_diagnosis.py` 做训练和评估
- 用 `checkpoint-best_auroc.pth` 做跨数据集评估
- 用 `--export_auroc_json` 额外导出可供 AUROC 绘图脚本直接读取的 JSON 结果
