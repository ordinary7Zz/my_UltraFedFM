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
import torchmetrics
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


def classification_specificity(y_true, y_pred, num_classes):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    total = cm.sum()
    specificities = []
    for class_idx in range(num_classes):
        tp = cm[class_idx, class_idx]
        fp = cm[:, class_idx].sum() - tp
        fn = cm[class_idx, :].sum() - tp
        tn = total - tp - fp - fn
        denom = tn + fp
        if denom > 0:
            specificities.append(tn / denom)
    if len(specificities) == 0:
        return np.nan
    return float(np.mean(specificities))


def classification_metric_ci(y_true, y_pred, y_score, num_classes):
    n = len(y_true)
    average_mode = 'binary' if num_classes == 2 else 'macro'

    def metric_from_bootstrap(metric_name, idx):
        y_true_i = y_true[idx]
        y_pred_i = y_pred[idx]
        y_score_i = y_score[idx]

        if metric_name == 'acc':
            return accuracy_score(y_true_i, y_pred_i)
        if metric_name == 'precision':
            return precision_score(y_true_i, y_pred_i, average=average_mode, zero_division=0)
        if metric_name == 'recall':
            return recall_score(y_true_i, y_pred_i, average=average_mode, zero_division=0)
        if metric_name == 'f1':
            return f1_score(y_true_i, y_pred_i, average=average_mode, zero_division=0)
        if metric_name == 'specificity':
            return classification_specificity(y_true_i, y_pred_i, num_classes)
        if np.unique(y_true_i).size < 2:
            return None
        if metric_name == 'auroc':
            if num_classes == 2:
                return roc_auc_score(y_true_i, y_score_i[:, 1])
            y_true_bin = label_binarize(y_true_i, classes=np.arange(num_classes))
            return roc_auc_score(y_true_bin, y_score_i, average='macro', multi_class='ovr')
        if metric_name == 'auprc':
            if num_classes == 2:
                return average_precision_score(y_true_i, y_score_i[:, 1])
            y_true_bin = label_binarize(y_true_i, classes=np.arange(num_classes))
            return average_precision_score(y_true_bin, y_score_i, average='macro')
        return None

    return {
        'acc_ci': bootstrap_ci(lambda idx: metric_from_bootstrap('acc', idx), n),
        'precision_ci': bootstrap_ci(lambda idx: metric_from_bootstrap('precision', idx), n),
        'recall_ci': bootstrap_ci(lambda idx: metric_from_bootstrap('recall', idx), n),
        'f1_ci': bootstrap_ci(lambda idx: metric_from_bootstrap('f1', idx), n),
        'specificity_ci': bootstrap_ci(lambda idx: metric_from_bootstrap('specificity', idx), n),
        'auroc_ci': bootstrap_ci(lambda idx: metric_from_bootstrap('auroc', idx), n),
        'auprc_ci': bootstrap_ci(lambda idx: metric_from_bootstrap('auprc', idx), n),
    }


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

    test_acc = torchmetrics.Accuracy('multiclass', average='micro', num_classes=args.nb_classes).cuda()
    test_recall = torchmetrics.Recall('multiclass', average='macro', num_classes=args.nb_classes).cuda()
    test_precision = torchmetrics.Precision('multiclass', average='macro', num_classes=args.nb_classes).cuda()
    test_f1 = torchmetrics.F1Score('multiclass', average='macro', num_classes=args.nb_classes).cuda()
    test_auroc = torchmetrics.AUROC("multiclass", average='macro', num_classes=args.nb_classes).cuda()
    test_aupr = torchmetrics.AveragePrecision("multiclass", average='macro', num_classes=args.nb_classes).cuda()
    test_spe = torchmetrics.Specificity('multiclass', average='macro', num_classes=args.nb_classes).cuda()
    test_cm = torchmetrics.ConfusionMatrix('multiclass', num_classes=args.nb_classes).cuda()
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

        test_acc(output.argmax(1), target)
        test_recall(output.argmax(1), target)
        test_precision(output.argmax(1), target)
        test_f1(output.argmax(1), target)
        test_spe(output.argmax(1), target)
        test_auroc(output, target)
        test_aupr(output, target)
        test_cm(output, target)

    cm = ConfusionMatrix(actual_vector=true_label_decode_list, predict_vector=prediction_decode_list)
    y_true_np = np.asarray(true_label_decode_list)
    y_pred_np = np.asarray(prediction_decode_list)
    y_score_np = np.asarray(prediction_score_list)
    ci_stats = classification_metric_ci(y_true_np, y_pred_np, y_score_np, args.nb_classes)
    if args.nb_classes > 2:
        total_acc = test_acc.compute()
    else:
        total_acc = cm.ACC_Macro
    total_recall = test_recall.compute()
    total_precision = test_precision.compute()
    total_auroc = test_auroc.compute()
    total_aupr = test_aupr.compute()
    total_f1 = test_f1.compute()
    total_spe = test_spe.compute()
    total_cm = test_cm.compute()
    total_loss = total_loss / count

    summary_message = (
        'TEST Epoch:{epoch} * ACC {acc:.3f} Precision {prec:.3f} Recall {rec:.3f} '
        'F1 {f1:.3f} AUROC {auroc:.3f} AUPR {aupr:.3f} SPE {spe:.3f} Loss {loss:.3f} '
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
        'ACC [{acc_lo:.3f}, {acc_hi:.3f}] '
        'Precision [{prec_lo:.3f}, {prec_hi:.3f}] '
        'Recall [{rec_lo:.3f}, {rec_hi:.3f}] '
        'F1 [{f1_lo:.3f}, {f1_hi:.3f}] '
        'AUROC [{auroc_lo:.3f}, {auroc_hi:.3f}] '
        'AUPRC [{auprc_lo:.3f}, {auprc_hi:.3f}] '
        'SPE [{spe_lo:.3f}, {spe_hi:.3f}]'
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
    }
