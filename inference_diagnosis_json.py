import os
import json
import csv
import torch
import logging
import argparse
import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn

from pycm import ConfusionMatrix
from torch.utils.data import DataLoader
from torchvision import datasets

import models_vit
from util.datasets import build_transform


class EvalImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target, path


def sanitize_name(name):
    if name is None:
        return 'dataset'
    return name.replace('\\', '_').replace('/', '_').replace(' ', '_')


def infer_dataset_name(data_path):
    normalized_path = os.path.normpath(data_path)
    base_name = os.path.basename(normalized_path)
    if base_name.lower() == 'test':
        base_name = os.path.basename(os.path.dirname(normalized_path))
    return sanitize_name(base_name)


def resolve_eval_root(data_path):
    test_root = os.path.join(data_path, 'test')
    if os.path.isdir(test_root):
        return test_root
    return data_path


def build_auroc_sample_records(test_stats, args):
    if args.nb_classes != 2:
        raise ValueError('This inference script only supports AUROC JSON export for binary classification.')

    y_true = test_stats['y_true']
    y_pred = test_stats['y_pred']
    y_score = test_stats['y_score']
    image_paths = test_stats.get('image_paths', [])
    has_image_paths = len(image_paths) == len(y_true)
    records = []

    for idx, (true_label, predicted_class, score_row) in enumerate(zip(y_true, y_pred, y_score)):
        prob_class_0 = float(score_row[0])
        prob_class_1 = float(score_row[1])
        confidence = float(score_row[int(predicted_class)])
        record = {
            'record_type': 'sample',
            'selected_model': args.model,
            'true_label': int(true_label),
            'predicted_class': int(predicted_class),
            'confidence': confidence,
            'prob_class_0': prob_class_0,
            'prob_class_1': prob_class_1,
        }
        if has_image_paths:
            image_file = os.path.abspath(image_paths[idx])
            record['image_file'] = image_file
            record['image_name'] = os.path.basename(image_file)
        records.append(record)

    return records


def evaluate_for_inference(data_loader, model, device, nb_classes):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    total_loss = 0.0
    count = 0
    y_true = []
    y_pred = []
    y_score = []
    image_paths = []

    for batch in data_loader:
        images, target, batch_paths = batch
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.no_grad():
            output = model(images)
            loss = criterion(output, target)
            probabilities = torch.softmax(output, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

        batch_size = images.shape[0]
        total_loss += loss.item() * batch_size
        count += batch_size
        y_true.extend(target.cpu().tolist())
        y_pred.extend(predictions.cpu().tolist())
        y_score.extend(probabilities.cpu().tolist())
        image_paths.extend(list(batch_paths))

    if count == 0:
        raise ValueError('Evaluation dataset is empty.')

    return {
        'loss': total_loss / count,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_score': y_score,
        'image_paths': image_paths,
    }


def load_checkpoint_model(model, resume_path, nb_classes):
    checkpoint = torch.load(resume_path, map_location='cpu')
    checkpoint_model = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint

    head_weight = checkpoint_model.get('head.weight') if isinstance(checkpoint_model, dict) else None
    if head_weight is not None and head_weight.shape[0] != nb_classes:
        raise ValueError(
            f'Checkpoint class count mismatch: checkpoint has {head_weight.shape[0]} classes, '
            f'but --nb_classes is {nb_classes}.'
        )

    model.load_state_dict(checkpoint_model, strict=True)
    return checkpoint


def get_args_parser():
    parser = argparse.ArgumentParser('UltraFedFM standalone diagnosis inference with AUROC JSON export')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--model', default='vit_large_patch16', type=str)
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--drop_path', type=float, default=0.1)
    parser.add_argument('--data_path', required=True, type=str,
                        help='Dataset root. Supports either <root>/test/<class>/... or <root>/<class>/...')
    parser.add_argument('--nb_classes', default=2, type=int)
    parser.add_argument('--resume', required=True, type=str)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool')
    parser.add_argument('--eval', action='store_true',
                        help='Compatibility flag. This standalone script always runs in eval mode.')
    parser.add_argument('--export_auroc_json', action='store_true',
                        help='Compatibility flag. AUROC JSON export is always enabled in this script.')
    parser.add_argument('--export_json_name', default='auroc_results.json', type=str)
    return parser


def main(args):
    if args.nb_classes != 2:
        raise ValueError('This standalone inference script currently supports only binary classification.')

    eval_root = resolve_eval_root(args.data_path)
    dataset_name = infer_dataset_name(args.data_path)
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    args.log_dir = os.path.join(os.path.dirname(os.path.abspath(args.resume)), f'eval_{dataset_name}_{timestamp}')
    os.makedirs(args.log_dir, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(args.log_dir, 'logging.log'),
        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
        level=logging.INFO,
        filemode='a',
        datefmt='%Y-%m-%d %I:%M:%S %p'
    )

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print('{}'.format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True

    transform = build_transform(is_train=False, args=args)
    dataset_val = EvalImageFolder(eval_root, transform=transform)
    print(dataset_val)
    print(f'class_to_idx: {dataset_val.class_to_idx}')
    logging.info('Eval root: %s', eval_root)
    logging.info('class_to_idx: %s', dataset_val.class_to_idx)

    data_loader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        shuffle=False,
    )

    model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )
    load_checkpoint_model(model, args.resume, args.nb_classes)
    print('Resume checkpoint {}'.format(args.resume))
    logging.info('Resume checkpoint %s', args.resume)
    model.to(device)

    test_stats = evaluate_for_inference(data_loader_val, model, device, args.nb_classes)
    cm = ConfusionMatrix(actual_vector=test_stats['y_true'], predict_vector=test_stats['y_pred'])

    with open(os.path.join(args.log_dir, 'roc.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for true_label, predicted_label in zip(test_stats['y_true'], test_stats['y_pred']):
            writer.writerow([true_label, predicted_label])

    cm.save_csv(os.path.join(args.log_dir, 'overall_stat.csv'))
    cm.plot(cmap=plt.cm.Blues, number_label=True, normalized=True, plot_lib='matplotlib')
    plt.savefig(os.path.join(args.log_dir, 'confusion_matrix.jpg'), dpi=600, bbox_inches='tight')
    plt.close()

    sample_records = build_auroc_sample_records(test_stats, args)
    export_path = os.path.join(args.log_dir, args.export_json_name)
    with open(export_path, 'w', encoding='utf-8') as f:
        json.dump(sample_records, f, ensure_ascii=False, indent=2)
    logging.info('Saved AUROC JSON to %s', export_path)
    print('Saved AUROC JSON to {}'.format(export_path))


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
