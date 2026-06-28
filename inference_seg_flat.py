import os
import cv2
import torch
import argparse
import numpy as np
import torch.nn.functional as F
import albumentations as A

from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2

import segmentation_models_pytorch as smp


SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


class FlatImageDataset(Dataset):
    """Load all images from a flat directory (no subdirectories, no labels)."""

    def __init__(self, root, img_size=224):
        self.root = root
        self.img_size = img_size
        self.image_paths = []
        self.image_names = []
        for fname in sorted(os.listdir(root)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_EXTS:
                self.image_paths.append(os.path.join(root, fname))
                self.image_names.append(fname)

        # Match original ValData preprocessing: Normalize → Resize → ToTensor
        self.transform = A.Compose([
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            A.Resize(img_size, img_size),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        path = self.image_paths[index]
        name = self.image_names[index]

        # Match original: cv2.imread → BGR2RGB
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]

        image = self.transform(image=image)['image']
        return image, name, orig_h, orig_w


def get_args_parser():
    parser = argparse.ArgumentParser('UltraFedFM binary segmentation flat-directory inference')

    parser.add_argument('--data_path', required=True, type=str,
                        help='Flat directory containing images (no subdirectories)')
    parser.add_argument('--resume', required=True, type=str,
                        help='Path to segmentation checkpoint .pth file')
    parser.add_argument('--output_dir', required=True, type=str,
                        help='Output directory for predicted masks')
    parser.add_argument('--img_size', default=224, type=int,
                        help='Input image size (default: 224)')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threshold', default=0.5, type=float,
                        help='Binarization threshold (default: 0.5)')

    return parser


@torch.no_grad()
def run_inference(data_loader, model, device, threshold, output_dir):
    model.eval()
    for images, names, orig_hs, orig_ws in data_loader:
        images = images.to(device, non_blocking=True)
        outputs = model(images)  # (B, 1, H, W), activation='sigmoid' already applied

        for i in range(images.size(0)):
            name = names[i]
            orig_h, orig_w = orig_hs[i].item(), orig_ws[i].item()

            pred = outputs[i]  # (1, H, W)
            pred = F.interpolate(pred.unsqueeze(0), size=(orig_h, orig_w), mode='bilinear', align_corners=False)
            pred = pred.squeeze()  # (orig_H, orig_W)
            mask = (pred > threshold).cpu().numpy().astype(np.uint8) * 255

            out_name = os.path.splitext(name)[0] + '.png'
            out_path = os.path.join(output_dir, out_name)
            cv2.imwrite(out_path, mask)

    print('Saved {} masks to {}'.format(len(data_loader.dataset), output_dir))


def main():
    args = get_args_parser().parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    # --- dataset ---
    dataset = FlatImageDataset(args.data_path, img_size=args.img_size)
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
    model = smp.Unet(
        encoder_name='mae',
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation='sigmoid',
    )
    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint)
    print('Loaded checkpoint from {}'.format(args.resume))
    model.to(device)

    # --- inference ---
    run_inference(data_loader, model, device, args.threshold, args.output_dir)


if __name__ == '__main__':
    main()
