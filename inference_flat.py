import os
import csv
import torch
import argparse
import numpy as np
import torch.backends.cudnn as cudnn

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import models_vit
from util.datasets import build_transform


SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


class FlatImageDataset(Dataset):
    """Load all images from a flat directory (no subdirectories, no labels)."""

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.image_paths = []
        for fname in sorted(os.listdir(root)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_EXTS:
                self.image_paths.append(os.path.join(root, fname))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        path = self.image_paths[index]
        image = Image.open(path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, path


def get_args_parser():
    parser = argparse.ArgumentParser('UltraFedFM flat-directory inference')

    parser.add_argument('--data_path', required=True, type=str,
                        help='Flat directory containing images (no subdirectories)')
    parser.add_argument('--resume', required=True, type=str,
                        help='Path to checkpoint .pth file')
    parser.add_argument('--nb_classes', default=2, type=int,
                        help='Number of classes')
    parser.add_argument('--output_csv', default=None, type=str,
                        help='Output CSV path (default: <checkpoint_dir>/predictions_<timestamp>.csv)')

    parser.add_argument('--model', default='vit_base_patch16', type=str)
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--global_pool', action='store_true', default=True)

    return parser


def load_model_from_checkpoint(model, resume_path):
    checkpoint = torch.load(resume_path, map_location='cpu')
    checkpoint_model = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
    model.load_state_dict(checkpoint_model, strict=True)
    print('Loaded checkpoint from {}'.format(resume_path))


@torch.no_grad()
def run_inference(data_loader, model, device):
    model.eval()
    all_paths = []
    all_preds = []
    all_confs = []

    for images, paths in data_loader:
        images = images.to(device, non_blocking=True)
        output = model(images)
        probabilities = torch.softmax(output, dim=1)
        preds = torch.argmax(probabilities, dim=1)
        confs = probabilities.gather(1, preds.unsqueeze(1)).squeeze(1)

        all_paths.extend(list(paths))
        all_preds.extend(preds.cpu().tolist())
        all_confs.extend(confs.cpu().tolist())

    return all_paths, all_preds, all_confs


def main():
    args = get_args_parser().parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True

    # --- output path ---
    if args.output_csv is None:
        import datetime
        checkpoint_dir = os.path.dirname(os.path.abspath(args.resume))
        data_name = os.path.basename(os.path.normpath(args.data_path))
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_csv = os.path.join(checkpoint_dir, 'predictions_{}_{}.csv'.format(data_name, timestamp))
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    # --- dataset ---
    transform = build_transform(is_train=False, args=args)
    dataset = FlatImageDataset(args.data_path, transform=transform)
    print('Found {} images in {}'.format(len(dataset), args.data_path))

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
    )

    # --- model ---
    model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=0.0,
        global_pool=args.global_pool,
    )
    load_model_from_checkpoint(model, args.resume)
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Model: {}, Params: {:.2f}M'.format(args.model, n_params / 1e6))

    # --- inference ---
    paths, preds, confs = run_inference(data_loader, model, device)

    # --- save CSV ---
    with open(args.output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'predicted_class', 'confidence'])
        for p, c, conf in zip(paths, preds, confs):
            writer.writerow([os.path.basename(p), c, '{:.6f}'.format(conf)])

    print('Saved {} predictions to {}'.format(len(paths), args.output_csv))


if __name__ == '__main__':
    main()
