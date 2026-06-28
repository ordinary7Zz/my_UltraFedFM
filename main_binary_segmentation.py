import os
import sys
import cv2
import time
import torch
import ctypes
import random
import logging 
import argparse
import numpy as np
import torch.nn as nn
import albumentations as A
import torch.nn.functional as F
import matplotlib.pyplot as plt

libgcc_s = ctypes.CDLL('libgcc_s.so.1')
sys.dont_write_bytecode = True
sys.path.insert(0, '../')

from tqdm import tqdm
from timm import create_model
from tabulate import tabulate
from datetime import datetime, timedelta
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from warmup_scheduler import GradualWarmupScheduler
from scipy.ndimage.morphology import distance_transform_edt as edt

import segmentation_models_pytorch as smp


def sanitize_name(name):
    if name is None:
        return 'dataset'
    return name.replace('\\', '_').replace('/', '_').replace(' ', '_')


def dataset_name_from_path(path):
    normalized = os.path.normpath(path)
    base = os.path.basename(normalized)
    if base in {'image', 'mask'}:
        base = os.path.basename(os.path.dirname(normalized))
    return sanitize_name(base)


def list_files_by_stem(directory):
    files = {}
    for name in os.listdir(directory):
        if name.startswith('.'):
            continue
        stem = os.path.splitext(name)[0]
        files[stem] = os.path.join(directory, name)
    return files


def build_matched_samples(image_dir, mask_dir):
    image_files = list_files_by_stem(image_dir)
    mask_files = list_files_by_stem(mask_dir)
    samples = sorted(set(image_files) & set(mask_files))
    return samples, image_files, mask_files


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

class TrainData(Dataset):
    def __init__(self, args, image_dir, mask_dir):
        self.args = args
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.samples, self.image_files, self.mask_files = build_matched_samples(image_dir, mask_dir)
        label_fraction = 1
        self.samples = random.sample(self.samples, int(len(self.samples) * label_fraction))
        self.transform = A.Compose([
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            A.Resize(args.img_size, args.img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            ToTensorV2()
        ])

    def __getitem__(self, idx):
        stem = self.samples[idx]
        image = cv2.imread(self.image_files[stem])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_files[stem], cv2.IMREAD_GRAYSCALE) / 255.0
        pair = self.transform(image=image, mask=mask)
        return pair['image'], pair['mask']

    def __len__(self):
        return len(self.samples)


class ValData(Dataset):
    def __init__(self, args, image_dir, mask_dir):
        self.args = args
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.samples, self.image_files, self.mask_files = build_matched_samples(image_dir, mask_dir)
        label_fraction = 1
        self.samples = random.sample(self.samples, int(len(self.samples) * label_fraction))
        self.img_transform = A.Compose([
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            A.Resize(args.img_size, args.img_size),
            ToTensorV2()
        ])
        self.mask_transform = A.Compose([
            A.Resize(args.img_size, args.img_size),
            ToTensorV2()
        ])

    def __getitem__(self, idx):
        stem = self.samples[idx]
        image = cv2.imread(self.image_files[stem])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_files[stem], cv2.IMREAD_GRAYSCALE) / 255.0
        image = self.img_transform(image=image)['image']
        mask = self.mask_transform(image=mask)['image']
        return image, mask, os.path.basename(self.image_files[stem])

    def __len__(self):
        return len(self.samples)

def bce_dice(pred, mask):
    ce_loss   = F.binary_cross_entropy_with_logits(pred, mask)
    # pred      = torch.sigmoid(pred)
    inter     = (pred*mask).sum(dim=(1,2))
    union     = pred.sum(dim=(1,2))+mask.sum(dim=(1,2))
    dice_loss = 1-(2*inter/(union+1)).mean()
    return ce_loss, dice_loss

class HausdorffDistance:
    def hd_distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        # if not np.any(x):
        #     x[0][0] = 1.0
        # elif not np.any(y):
        #     y[0][0] = 1.0

        indexes = np.nonzero(x)
        distances = edt(np.logical_not(y))

        ###modified here###        
        # 如果没有非零元素，返回0
        if len(indexes[0]) == 0:
            return np.array(0.0)

        return np.array(np.percentile(distances[indexes], 95))

    def compute(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert (
            pred.shape[1] == 1 and target.shape[1] == 1
            ), "Only binary channel supported"

        pred = (pred > 0.5).byte()
        target = (target > 0.5).byte()
        if torch.sum(pred) == 0:
            pred[0][0][0][0] = 1
            # print(pred)
            # print(torch.sum(pred))
        # print(pred.shape)
        right_hd = torch.from_numpy(
            self.hd_distance(pred.cpu().numpy(), target.cpu().numpy())
            ).float()

        left_hd = torch.from_numpy(
            self.hd_distance(target.cpu().numpy(), pred.cpu().numpy())
            ).float()


        return torch.max(right_hd, left_hd)

def evaluate(pred, gt):
    if isinstance(pred, (list, tuple)):
        pred = pred[0]

    pred_binary = (pred >= 0.5).float()
    pred_binary_inverse = (pred_binary == 0).float()

    gt_binary = (gt >= 0.5).float()
    gt_binary_inverse = (gt_binary == 0).float()

    TP = pred_binary.mul(gt_binary).sum()
    FP = pred_binary.mul(gt_binary_inverse).sum()
    FN = pred_binary_inverse.mul(gt_binary).sum()

    if TP.item() == 0:
        TP = torch.tensor(1.0, device=pred.device)

    # IoU
    IoU = TP / (TP + FP + FN)
    # DICE
    DICE = 2 * IoU / (IoU + 1)

    pred  = pred.data.cpu().numpy().squeeze()
    gt    = gt.data.cpu().numpy().squeeze()
    gt    /= (gt.max() + 1e-8)
    pred  = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    mae   = np.sum(np.abs(pred-gt))*1.0/(gt.shape[0]*gt.shape[1])



    
    return IoU.item(), DICE.item(), float(mae)


class Train(object):
    def __init__(self, TrainData, ValData, args):
        ## dataset
        self.args      = args 
        # Only load training data if not in eval/plot mode
        if not args.eval and not args.plot and not args.eval_instance:
            self.train_data    = TrainData(args, args.train_image_path, args.train_mask_path)
            self.train_loader  = DataLoader(self.train_data, batch_size=int(args.batch_size), pin_memory=True, shuffle=True, num_workers=args.num_workers)
            print('train dataset: ', len(self.train_data))
        else:
            self.train_data = None
            self.train_loader = None

        self.val_data      = ValData(args, args.test_image_path, args.test_mask_path)
        self.val_loader    = DataLoader(self.val_data, batch_size=1, pin_memory=True, shuffle=True, num_workers=args.num_workers)
        print('val dataset: ', len(self.val_data))
        ## model
        ENCODER = 'mae'
        ENCODER_WEIGHTS = args.pretrained
        ACTIVATION = 'sigmoid'
        self.model = smp.Unet(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS, 
                                 in_channels=3, classes=1, activation=ACTIVATION)
        print('load pretrained weight from {}'.format(args.pretrained))
        logging.info('load pretrained weight from {}'.format(args.pretrained))

        if args.resume:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume, map_location='cpu')
                self.model.load_state_dict(checkpoint)
        # self.model.train(True)
        self.model.cuda()
        ## parameter
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epoch, eta_min=1e-6)
        warmup_epochs  = args.epoch // 10
        self.scheduler = GradualWarmupScheduler(self.optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=self.scheduler)
        self.scheduler.step()
        # self.model, self.optimizer = apex.amp.initialize(self.model, self.optimizer, opt_level='O2')
        if not args.eval and not args.plot:
            self.logger    = SummaryWriter(args.exp_path)
        self.best_mae   = 1
        self.best_dice  = 0
        self.best_acc   = 0
        self.best_epoch = 0
        self.best_f1    = 0

    def train(self):
        global_step = 0
        EARLY_STOPS = 100
        for epoch in range(self.args.epoch):
            self.model.train()
            total_loss_ce = 0.0
            total_loss_dice = 0.0
            total_loss = 0.0
            num_batches = 0

            for image, mask in self.train_loader:
                image, mask = image.cuda().float(), mask.cuda().float()

                pred = self.model(image)
                pred = F.interpolate(pred, size=mask.shape[1:], mode='bilinear', align_corners=True)[:,0,:,:]
                loss_ce, loss_dice = bce_dice(pred, mask)

                self.optimizer.zero_grad()
                loss = loss_ce + loss_dice
                loss.backward()
                self.optimizer.step()

                ## log (tensorboard only, no console print)
                global_step += 1
                self.logger.add_scalar('lr', self.optimizer.param_groups[0]['lr'], global_step=global_step)
                self.logger.add_scalars('loss', {'ce': loss_ce.item(), 'dice': loss_dice.item()}, global_step=global_step)

                total_loss_ce += loss_ce.item()
                total_loss_dice += loss_dice.item()
                total_loss += loss.item()
                num_batches += 1

            self.scheduler.step()

            avg_ce = total_loss_ce / num_batches
            avg_dice = total_loss_dice / num_batches
            avg_loss = total_loss / num_batches
            print(f'{datetime.now()} | epoch: {epoch+1:d}/{self.args.epoch:d} | lr={self.optimizer.param_groups[0]["lr"]:.6f} | ce={avg_ce:.6f} | dice={avg_dice:.6f} | loss={avg_loss:.6f}')
            logging.info(f'{datetime.now()} | epoch: {epoch+1:d}/{self.args.epoch:d} | lr={self.optimizer.param_groups[0]["lr"]:.6f} | ce={avg_ce:.6f} | dice={avg_dice:.6f} | loss={avg_loss:.6f}')

            self.val(self.val_loader, self.model, epoch, self.args.exp_path)

            # if (epoch+1)%8==0:
            #     torch.save(self.model.state_dict(), self.args.savepath+'/model-'+str(epoch+1))

    def val(self, val_loader, model, epoch, save_path):
        # best_mae, best_dice, best_acc, best_epoch
        model.eval()
        with torch.no_grad():
            mae_sum  = 0
            iou_sum  = 0
            dice_sum = 0
            sen_sum  = 0
            spe_sum = 0
            acc_sum = 0
            seconds = 0
            dice_lst = []
            iou_lst = []
            mae_lst = []
            hd_lst = []
            hd_metric = HausdorffDistance()

            for image, mask, _ in tqdm(val_loader, total=len(val_loader), desc='Validation'):
                image    = image.cuda()
                mask     = mask.cuda()

                start     = time.time()
                pred      = model(image)
                end       = time.time()
                seconds += end - start
  
                iou, dice, mae  = evaluate(pred, mask)
                hd = hd_metric.compute(pred, mask)
                hd = hd.numpy()
                dice_lst.append(dice)
                iou_lst.append(iou)
                mae_lst.append(mae)
                hd_lst.append(hd)
                mask_pred_show = (pred.squeeze().cpu().numpy())*255

            fps     = len(val_loader) / seconds

            dice = np.average(dice_lst)
            iou = np.average(iou_lst)
            mae = np.average(mae_lst)
            hd = np.average(hd_lst)
            if type(dice) is np.ndarray:
                dice = dice[0]
            if type(iou) is np.ndarray:
                iou = iou[0]
            if type(mae) is np.ndarray:
                mae = mae[0]
            if type(hd) is np.ndarray:
                hd = hd[0]
            self.logger.add_scalar('MAE', mae, global_step=epoch)
            self.logger.add_scalar('I0U', iou, global_step=epoch)
            self.logger.add_scalar('Dice', dice, global_step=epoch)
            # self.logger.add_scalar('HD', hd, global_step=epoch)
            # if mae < self.best_mae:
            #     self.best_mae   = mae
            #     self.best_epoch = epoch
            #     # torch.save(model.state_dict(), save_path+'/epoch_bestMAE.pth')
            #     print(f'best MAE {self.best_mae:.3f} epoch:{epoch}')
            #     logging.info(f'best MAE {self.best_mae:.3f} epoch:{epoch}')
                
            if dice > self.best_dice:
                self.best_dice   = dice
                self.best_epoch = epoch
                torch.save(model.state_dict(), save_path+'/epoch_bestDice.pth')
                print(f'best Dice {self.best_dice:.3f} (IOU: {iou:.3f}) epoch:{epoch}')
                logging.info(f'best Dice {self.best_dice:.3f} (IOU: {iou:.3f}) epoch:{epoch}')
                    
            print(f'#TEST#:  MAE: {mae:.3f}  IoU: {iou:.3f} Dice: {dice:.3f}  fps: {fps:.3f} ####   bestDice: {self.best_dice:.3f}')
            logging.info(f'#TEST#: MAE: {mae:.3f}  IoU: {iou:.3f} Dice: {dice:.3f} fps: {fps:.3f} ####  bestDice: {self.best_dice:.3f}')
    
    def eval(self, val_loader, model, save_path):
        model.eval()
        with torch.no_grad():
            dice_list = []
            hd95_list = []
            mae_list = []
            iou_list = []
            hd_metric = HausdorffDistance()
            with open(save_path+'/eval.txt', 'w') as f:
                for image, mask, name in tqdm(val_loader, total=len(val_loader), desc='Validation'):
                    image    = image.cuda()
                    mask     = mask.cuda()
                    pred      = model(image)
                    iou, dice, mae  = evaluate(pred, mask)
                    hd = hd_metric.compute(pred, mask).item()

                    iou_list.append(float(iou))
                    dice_list.append(float(dice))
                    mae_list.append(float(mae))
                    hd95_list.append(float(hd))

                IoU = float(np.mean(iou_list))
                Dice = float(np.mean(dice_list))
                Mae = float(np.mean(mae_list))
                HD = float(np.mean(hd95_list))

                dice_arr = np.asarray(dice_list, dtype=float)
                hd95_arr = np.asarray(hd95_list, dtype=float)
                dice_mean, dice_lo, dice_hi = bootstrap_ci(lambda idx: dice_arr[idx].mean(), len(dice_arr))
                hd95_mean, hd95_lo, hd95_hi = bootstrap_ci(lambda idx: hd95_arr[idx].mean(), len(hd95_arr))

                summary_message = f'MAE: {Mae} HD95: {HD} IoU: {IoU} Dice: {Dice}'
                ci_message = f'95% CI | HD95 [{hd95_lo}, {hd95_hi}] Dice [{dice_lo}, {dice_hi}]'
                print(summary_message)
                print(ci_message)
                f.write(summary_message + '\n')
                f.write(ci_message + '\n')
                logging.info(summary_message)
                logging.info(ci_message)
 
    def eval_instance(self, val_loader, model, save_path):
        model.eval()
        with torch.no_grad():
            hd_metric = HausdorffDistance()
            with open(save_path+'/eval.txt', 'w') as f:
                for image, mask, name in tqdm(val_loader, total=len(val_loader), desc='Validation'):
                    image    = image.cuda()
                    mask     = mask.cuda()
                    pred      = model(image)
                    iou, dice, mae  = evaluate(pred, mask)
                    hd = hd_metric.compute(pred, mask)
                    hd = hd.numpy()
                    line = f'Image name: {name}:  MAE: {mae} HD: {hd} IoU: {iou} Dice: {dice}' + '\n'
                    f.write(line)

    def plot(self, val_loader, model, save_path):
        model.eval()
        with torch.no_grad():
            for image, mask, name in tqdm(val_loader, total=len(val_loader), desc='Validation'):
                name = name[0]
                image    = image.cuda()
                mask     = mask.cuda()
                pred      = model(image)
                pred[pred < 0.5]=0
                pred[pred > 0.5]=1
                pred       = pred.squeeze().cpu().numpy()*255
                if not os.path.exists(os.path.join(save_path,'figures')):
                    os.makedirs(os.path.join(save_path,'figures'), exist_ok=True)
                cv2.imwrite(os.path.join(save_path,'figures/', name), np.uint8(pred))



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath'    , type=str,     default='../data/train'         )
    parser.add_argument('--train_image_path', type=str,  default=None                    )
    parser.add_argument('--train_mask_path', type=str,   default=None                    )
    parser.add_argument('--test_image_path', type=str,   default=None                    )
    parser.add_argument('--test_mask_path', type=str,    default=None                    )
    parser.add_argument('--savepath'    , type=str,     default='./out'                 )
    parser.add_argument('--model_name'  , type=str,     default='vit_base_patch16'      )
    parser.add_argument('--mode'        , type=str,     default='train'                 )
    parser.add_argument('--lr'          , type=float,   default=0.01                    )
    parser.add_argument('--img_size'    , type=int,     default=224                     )
    parser.add_argument('--epoch'       , type=int,     default=128                     )
    parser.add_argument('--batch_size'  , type=int,     default=2                       )
    parser.add_argument('--weight_decay', type=float,   default=5e-4                    )
    parser.add_argument('--momentum'    , type=float,   default=0.9                     )
    parser.add_argument('--nesterov'    , default=True                                  )
    parser.add_argument('--num_workers' , type=int,     default=4                       )
    parser.add_argument('--gpu_id'      , type=str,     default='1'                     )
    parser.add_argument('--pretrained'  , type=str,     default=None                    )
    parser.add_argument('--note'        , type=str,     default=None                    )
    parser.add_argument('--eval'        , action='store_true'                           )
    parser.add_argument('--eval_instance' , action='store_true'                         )
    parser.add_argument('--plot'        , action='store_true'                           )
    parser.add_argument('--resume'      , type=str,     default=None                    )
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    if args.plot or args.eval:
        save_path = os.path.join(args.savepath, args.note) if args.note else args.savepath
        dataset_source = args.note if args.note else (args.test_image_path or args.train_image_path or args.datapath)
        dataset_name = dataset_name_from_path(dataset_source)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        mode_prefix = 'plot' if args.plot else 'eval'
        args.exp_path = os.path.join('/'.join(args.resume.split('/')[:-1]), f'{mode_prefix}_{dataset_name}_{timestamp}')
        os.makedirs(args.exp_path, exist_ok=True)
    else:
        save_path          = os.path.join(args.savepath, args.note) if args.note else args.savepath
        current_timestamp  = datetime.now().timestamp()
        current_datetime   = datetime.fromtimestamp(current_timestamp+29220)  # different time zone
        formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H:%M:%S")
        args.exp_path      = os.path.join(save_path, 'log_'+formatted_datetime)

        os.makedirs(save_path, exist_ok=True)
        os.makedirs(args.exp_path, exist_ok=True)

    logging.basicConfig(filename=args.exp_path+'/log.log',format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', level = logging.INFO,filemode='a',datefmt='%Y-%m-%d %I:%M:%S %p')
    tables  = [[args.pretrained, save_path, args.lr, args.epoch, args.batch_size, args.weight_decay, args.note]]
    headers = ['pretrained''savepath', 'lr', 'epoch', 'batch_size', 'weight_decay', 'note']
    print('===training configures===')
    print(tabulate(tables, headers, tablefmt="grid", numalign="center"))
    logging.info('\n'+tabulate(tables, headers, tablefmt="github", numalign="center"))

    t    = Train(TrainData, ValData, args)
    if args.plot:
        print("Start svaing prediction results")
        t.plot(t.val_loader, t.model, args.exp_path)
    elif args.eval_instance:
        print("Start instance evaluating")
        t.eval_instance(t.val_loader, t.model, args.exp_path)
    elif args.eval:
        print("Start evaluating")
        t.eval(t.val_loader, t.model, args.exp_path)
    else:
        print("Start training")
        t.train()


