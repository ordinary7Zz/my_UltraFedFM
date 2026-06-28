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
- `scripts/eval_all_classification_json.sh`：调用独立推理脚本批量评估多个甲状腺分类数据集，并额外导出 AUROC JSON

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
- 类别目录名不强制必须叫 `benign` / `malignant`，也可以是 `0` / `1`
- 但二分类评估默认把 **class index 1** 当作正类概率 `prob_class_1`
- `ImageFolder` 会按**子目录名字典序**分配类别编号，因此要确保“恶性/正类”排在编号 `1`
- 最推荐的命名是：`benign -> 0`、`malignant -> 1`；如果用数字目录，也建议固定为 `0=benign`、`1=malignant`
- 训练仍然要求使用 `train/` + `test/` 结构
- **新的独立推理脚本**允许只传测试集：如果目录下存在 `test/`，就读取 `test/`；如果没有 `test/`，则直接把当前目录当成测试集根目录（例如只有 `0/1` 或 `benign/malignant` 子目录也可以）

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
为了给 `plot_single_task_auroc.py` 使用，现在提供一个**独立推理脚本**来导出标准 JSON。

### 单数据集导出方式

当前推荐入口：`inference_diagnosis_json.py`

```bash
CUDA_VISIBLE_DEVICES=0 python inference_diagnosis_json.py \
  --model vit_base_patch16 \
  --batch_size 16 \
  --nb_classes 2 \
  --data_path ./dataset/Classification/finall_data \
  --resume ./output_dir/dataset_3_cls_experiment/log_2026-02-27_18:47:44/checkpoint-best_auroc.pth
```

如果想自定义输出文件名，还可以加：

```bash
--export_json_name my_results.json
```

### 新推理脚本支持的目录形式

支持两种：

```text
方式 A：
<data_path>/test/<class_name>/*.png

方式 B：
<data_path>/<class_name>/*.png
```

也就是说：
- 如果目录下有 `test/`，脚本会自动读取 `test/`
- 如果目录下没有 `test/`，脚本会直接把当前目录当成测试集根目录

### 批量导出方式

当前新增脚本：`scripts/eval_all_classification_json.sh`

它与 `scripts/eval_all_classification.sh` 的评估参数保持一致，但底层改成调用独立推理脚本：

```bash
python inference_diagnosis_json.py ...
```

也就是说：
- 模型相同
- `batch_size` 相同
- `nb_classes` 相同
- `data_path` 相同
- `resume` 相同
- 评估的数据集列表相同
- 只是入口改成了专用推理脚本，并且每次评估目录下都会导出一个 JSON 文件

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

## 7. 无标签扁平目录推理

当只有一批无标签图片（所有图片在同一目录下，无子目录），需要直接获取每张图的预测类别时，使用 `inference_flat.py`。

### 使用方式

```bash
CUDA_VISIBLE_DEVICES=0 python inference_flat.py \
  --data_path /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/sample/images \
  --resume ./output_dir/dataset_3_cls_experiment/log_2026-02-27_18:47:44/checkpoint-best_auroc.pth \
  --nb_classes 2
  --output_csv ./BM.csv
```

### 参数说明

| 参数 | 必填 | 说明 |
|---|---|---|
| `--data_path` | 是 | 图片目录，所有图片直接放在该目录下 |
| `--resume` | 是 | 权重 `.pth` 文件路径 |
| `--nb_classes` | 否(默认2) | 分类数 |
| `--model` | 否 | 模型名，默认 `vit_base_patch16` |
| `--batch_size` | 否 | 批量大小，默认 16 |
| `--output_csv` | 否 | 输出路径，默认生成在 checkpoint 同级目录下 |

### 输出 CSV 格式

| 列名 | 内容 |
|---|---|
| `image_path` | 图片文件名（不含路径） |
| `predicted_class` | 预测类别编号 |
| `confidence` | 该预测类别的概率值 |

---

## 8. 推荐使用顺序
如果现在要复现当前流程，建议按下面顺序：

1. 准备二分类目录结构数据集
2. 运行 `scripts/my_class_pretrain.sh` 训练模型
3. 用 `scripts/eval_all_classification.sh` 做批量评估
4. 如果要保留可直接画 AUROC 的结果，运行 `scripts/eval_all_classification_json.sh`，或直接调用 `inference_diagnosis_json.py`
5. 如果要对无标签扁平目录做推理，使用 `inference_flat.py`

---

## 9. 一句话总结
当前项目里，甲状腺良恶性二分类的最新用法是：

- 用 `main_diagnosis.py` 做训练和常规评估
- 用 `checkpoint-best_auroc.pth` 做跨数据集评估
- 用 `inference_diagnosis_json.py` 或 `scripts/eval_all_classification_json.sh` 导出可供 AUROC 绘图脚本直接读取的 JSON 结果
- 用 `inference_flat.py` 对无标签图片目录做快速推理，输出 CSV
