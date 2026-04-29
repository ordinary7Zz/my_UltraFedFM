# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import sys
import math
import torch
import logging
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from pycm import *
from timm.data import Mixup
from timm.utils import accuracy
from torchvision import transforms
from PIL import Image, ImageFilter
from typing import Iterable, Optional
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import label_binarize

import util.misc as misc
import util.lr_sched as lr_sched
import util.metrics as metrics


def bootstrap_ci(stat_fn, n, n_boot=2000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    values = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        value = stat_fn(idx)
        if value is None or np.isnan(value):
            continue
        values.append(float(value))

    if len(values) == 0:
        return np.nan, np.nan, np.nan

    values = np.asarray(values, dtype=float)
    point_estimate = float(stat_fn(np.arange(n)))
    lower = float(np.percentile(values, 100 * alpha / 2))
    upper = float(np.percentile(values, 100 * (1 - alpha / 2)))
    return point_estimate, lower, upper


def binary_specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp = cm[0, 0], cm[0, 1]
    denom = tn + fp
    if denom == 0:
        return np.nan
    return float(tn / denom)


def classification_specificity(y_true, y_pred, num_classes):
    if num_classes == 2:
        return binary_specificity(y_true, y_pred)

    specificities = []
    for class_idx in range(num_classes):
        y_true_i = (y_true == class_idx).astype(int)
        y_pred_i = (y_pred == class_idx).astype(int)
        specificity = binary_specificity(y_true_i, y_pred_i)
        if not np.isnan(specificity):
            specificities.append(specificity)
    if len(specificities) == 0:
        return np.nan
    return float(np.mean(specificities))


def classification_metrics(y_true, y_pred, y_score, num_classes):
    metrics = {
        'acc': float(accuracy_score(y_true, y_pred)),
        'specificity': classification_specificity(y_true, y_pred, num_classes),
    }

    if num_classes == 2:
        metrics['precision'] = float(precision_score(y_true, y_pred, average='binary', zero_division=0))
        metrics['recall'] = float(recall_score(y_true, y_pred, average='binary', zero_division=0))
        metrics['f1'] = float(f1_score(y_true, y_pred, average='binary', zero_division=0))
        if np.unique(y_true).size < 2:
            metrics['auroc'] = np.nan
            metrics['aupr'] = np.nan
        else:
            metrics['auroc'] = float(roc_auc_score(y_true, y_score[:, 1]))
            metrics['aupr'] = float(average_precision_score(y_true, y_score[:, 1]))
        return metrics

    metrics['precision'] = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
    metrics['recall'] = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
    metrics['f1'] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    if np.unique(y_true).size < num_classes:
        metrics['auroc'] = np.nan
        metrics['aupr'] = np.nan
    else:
        y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))
        metrics['auroc'] = float(roc_auc_score(y_true_bin, y_score, average='macro', multi_class='ovr'))
        metrics['aupr'] = float(average_precision_score(y_true_bin, y_score, average='macro'))
    return metrics


def classification_metric_ci(y_true, y_pred, y_score, num_classes):
    n = len(y_true)

    def metric_from_bootstrap(metric_name, idx):
        metrics = classification_metrics(y_true[idx], y_pred[idx], y_score[idx], num_classes)
        if metric_name == 'auprc':
            return metrics['aupr']
        return metrics[metric_name]

    return {
        'acc_ci': bootstrap_ci(lambda idx: metric_from_bootstrap('acc', idx), n),
        'precision_ci': bootstrap_ci(lambda idx: metric_from_bootstrap('precision', idx), n),
        'recall_ci': bootstrap_ci(lambda idx: metric_from_bootstrap('recall', idx), n),
        'f1_ci': bootstrap_ci(lambda idx: metric_from_bootstrap('f1', idx), n),
        'specificity_ci': bootstrap_ci(lambda idx: metric_from_bootstrap('specificity', idx), n),
        'auroc_ci': bootstrap_ci(lambda idx: metric_from_bootstrap('auroc', idx), n),
        'auprc_ci': bootstrap_ci(lambda idx: metric_from_bootstrap('auprc', idx), n),
    }


def classwise_classification_metrics(y_true, y_pred, y_score, num_classes):
    class_metrics = []

    for class_idx in range(num_classes):
        y_true_i = (y_true == class_idx).astype(int)
        y_pred_i = (y_pred == class_idx).astype(int)
        metrics = {
            'accuracy': float(accuracy_score(y_true_i, y_pred_i)),
            'precision': float(precision_score(y_true_i, y_pred_i, zero_division=0)),
            'recall': float(recall_score(y_true_i, y_pred_i, zero_division=0)),
            'f1': float(f1_score(y_true_i, y_pred_i, zero_division=0)),
            'specificity': binary_specificity(y_true_i, y_pred_i),
        }
        if np.unique(y_true_i).size < 2:
            metrics['auroc'] = np.nan
            metrics['aupr'] = np.nan
        else:
            metrics['auroc'] = float(roc_auc_score(y_true_i, y_score[:, class_idx]))
            metrics['aupr'] = float(average_precision_score(y_true_i, y_score[:, class_idx]))
        class_metrics.append(metrics)

    return class_metrics


def classwise_classification_metric_ci(y_true, y_pred, y_score, num_classes):
    n = len(y_true)
    class_ci = []

    for class_idx in range(num_classes):
        def metric_from_bootstrap(metric_name, idx, current_class=class_idx):
            y_true_i = (y_true[idx] == current_class).astype(int)
            y_pred_i = (y_pred[idx] == current_class).astype(int)
            y_score_i = y_score[idx, current_class]

            if metric_name == 'accuracy':
                return accuracy_score(y_true_i, y_pred_i)
            if metric_name == 'precision':
                return precision_score(y_true_i, y_pred_i, zero_division=0)
            if metric_name == 'recall':
                return recall_score(y_true_i, y_pred_i, zero_division=0)
            if metric_name == 'f1':
                return f1_score(y_true_i, y_pred_i, zero_division=0)
            if metric_name == 'specificity':
                return binary_specificity(y_true_i, y_pred_i)
            if np.unique(y_true_i).size < 2:
                return None
            if metric_name == 'auroc':
                return roc_auc_score(y_true_i, y_score_i)
            if metric_name == 'auprc':
                return average_precision_score(y_true_i, y_score_i)
            return None

        class_ci.append({
            'accuracy_ci': bootstrap_ci(lambda idx, m='accuracy': metric_from_bootstrap(m, idx), n),
            'precision_ci': bootstrap_ci(lambda idx, m='precision': metric_from_bootstrap(m, idx), n),
            'recall_ci': bootstrap_ci(lambda idx, m='recall': metric_from_bootstrap(m, idx), n),
            'f1_ci': bootstrap_ci(lambda idx, m='f1': metric_from_bootstrap(m, idx), n),
            'specificity_ci': bootstrap_ci(lambda idx, m='specificity': metric_from_bootstrap(m, idx), n),
            'auroc_ci': bootstrap_ci(lambda idx, m='auroc': metric_from_bootstrap(m, idx), n),
            'auprc_ci': bootstrap_ci(lambda idx, m='auprc': metric_from_bootstrap(m, idx), n),
        })

    return class_ci


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, epoch, logging, args):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    model.eval()

    total_loss = 0.
    count = 0

    prediction_decode_list = []
    true_label_decode_list = []
    prediction_score_list = []
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        true_label = F.one_hot(target.to(torch.int64), num_classes=args.nb_classes)
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

            prediction_softmax = nn.Softmax(dim=1)(output)
            _, prediction_decode = torch.max(prediction_softmax, 1)
            _, true_label_decode = torch.max(true_label, 1)

            prediction_decode_list.extend(prediction_decode.cpu().detach().numpy())
            true_label_decode_list.extend(true_label_decode.cpu().detach().numpy())
            prediction_score_list.extend(prediction_softmax.cpu().detach().numpy())
        batch_size = images.shape[0]
        total_loss += loss.item() * batch_size
        count += batch_size

    y_true_np = np.asarray(true_label_decode_list)
    y_pred_np = np.asarray(prediction_decode_list)
    y_score_np = np.asarray(prediction_score_list)
    metrics = classification_metrics(y_true_np, y_pred_np, y_score_np, args.nb_classes)
    ci_stats = classification_metric_ci(y_true_np, y_pred_np, y_score_np, args.nb_classes)
    classwise_metrics = classwise_classification_metrics(y_true_np, y_pred_np, y_score_np, args.nb_classes)
    classwise_ci_stats = classwise_classification_metric_ci(y_true_np, y_pred_np, y_score_np, args.nb_classes)
    total_cm = torch.tensor(confusion_matrix(y_true_np, y_pred_np, labels=list(range(args.nb_classes))), device=device)
    total_loss = total_loss / count

    total_acc = metrics['acc']
    total_precision = metrics['precision']
    total_recall = metrics['recall']
    total_f1 = metrics['f1']
    total_auroc = metrics['auroc']
    total_aupr = metrics['aupr']
    total_spe = metrics['specificity']

    summary_message = (
        'TEST Epoch:{epoch} * ACC {acc:.4f} Precision {prec:.4f} Recall {rec:.4f} '
        'F1 {f1:.4f} AUROC {auroc:.4f} AUPR {aupr:.4f} SPE {spe:.4f} Loss {loss:.4f} '
        '\n Confusion Matrix \n {cm}'
    ).format(
        epoch=epoch,
        acc=100 * total_acc,
        prec=total_precision,
        rec=total_recall,
        f1=total_f1,
        auroc=total_auroc,
        aupr=total_aupr,
        spe=total_spe,
        loss=total_loss,
        cm=total_cm,
    )
    print(summary_message)
    logging.info(summary_message)

    ci_message = (
        '95% CI | '
        'ACC [{acc_lo:.4f}, {acc_hi:.4f}] '
        'Precision [{prec_lo:.4f}, {prec_hi:.4f}] '
        'Recall [{rec_lo:.4f}, {rec_hi:.4f}] '
        'F1 [{f1_lo:.4f}, {f1_hi:.4f}] '
        'AUROC [{auroc_lo:.4f}, {auroc_hi:.4f}] '
        'AUPRC [{auprc_lo:.4f}, {auprc_hi:.4f}] '
        'SPE [{spe_lo:.4f}, {spe_hi:.4f}]'
    ).format(
        acc_lo=100 * ci_stats['acc_ci'][1], acc_hi=100 * ci_stats['acc_ci'][2],
        prec_lo=ci_stats['precision_ci'][1], prec_hi=ci_stats['precision_ci'][2],
        rec_lo=ci_stats['recall_ci'][1], rec_hi=ci_stats['recall_ci'][2],
        f1_lo=ci_stats['f1_ci'][1], f1_hi=ci_stats['f1_ci'][2],
        auroc_lo=ci_stats['auroc_ci'][1], auroc_hi=ci_stats['auroc_ci'][2],
        auprc_lo=ci_stats['auprc_ci'][1], auprc_hi=ci_stats['auprc_ci'][2],
        spe_lo=ci_stats['specificity_ci'][1], spe_hi=ci_stats['specificity_ci'][2],
    )
    print(ci_message)
    logging.info(ci_message)

    classwise_summary = []
    for class_idx, (class_metric, class_ci) in enumerate(zip(classwise_metrics, classwise_ci_stats)):
        classwise_summary.append({
            'Class': class_idx,
            'Accuracy': class_metric['accuracy'],
            'Precision': class_metric['precision'],
            'Recall': class_metric['recall'],
            'F1': class_metric['f1'],
            'AUROC': class_metric['auroc'],
            'AUPR': class_metric['aupr'],
            'SPE': class_metric['specificity'],
            'Accuracy 95% CI': class_ci['accuracy_ci'][1:],
            'Precision 95% CI': class_ci['precision_ci'][1:],
            'Recall 95% CI': class_ci['recall_ci'][1:],
            'F1 95% CI': class_ci['f1_ci'][1:],
            'AUROC 95% CI': class_ci['auroc_ci'][1:],
            'AUPR 95% CI': class_ci['auprc_ci'][1:],
            'SPE 95% CI': class_ci['specificity_ci'][1:],
        })

    return {
        'acc': 100 * total_acc,
        'precision': total_precision,
        'recall': total_recall,
        'f1': total_f1,
        'auroc': total_auroc,
        'aupr': total_aupr,
        'spe': total_spe,
        'loss': total_loss,
        'cm': total_cm,
        'y_true': true_label_decode_list,
        'y_pred': prediction_decode_list,
        'y_score': prediction_score_list,
        'ci': ci_stats,
        'metrics_consistent': metrics,
        'classwise_metrics_consistent': classwise_metrics,
        'classwise_ci_consistent': classwise_ci_stats,
        'classwise_summary': classwise_summary,
    }
