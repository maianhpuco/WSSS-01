from utils.validate import validate, generate_cam
from utils.hierarchical_utils import pair_features, merge_to_parent_predictions, merge_subclass_cams_to_parent, expand_parent_to_subclass_labels
from utils.contrast_loss import InfoNCELossFG, InfoNCELossBG
from utils.optimizer import PolyWarmupAdamW
from utils.fgbg_feature import FeatureExtractor, MaskAdapter_DynamicThreshold
from model.model import ClsNetwork
from model.model_ViLa_MIL import ViLa_MIL_Model
from model.projector import PLIPProjector
import argparse
import datetime
import os
import sys
import numpy as np
import cv2 as cv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from importlib.machinery import SourceFileLoader
import csv
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchmetrics import JaccardIndex
from omegaconf import OmegaConf
from tqdm import tqdm
import ttach as tta
from skimage import morphology
import time
import psutil
import h5py
import zipfile
from PIL import Image

# Add project root to sys.path
project_root = r"D:\NghienCuu\NghienCuuPaper\Source_Code\WSSS-01"
sys.path.append(project_root)

# Ensure model and utils are valid Python packages
model_dir = os.path.join(
    project_root, "src", "modified_externals", "ViLa-PIP", "model")
utils_dir = os.path.join(
    project_root, "src", "modified_externals", "ViLa-PIP", "utils")
for d in [model_dir, utils_dir]:
    if os.path.exists(d) and not os.path.exists(os.path.join(d, "__init__.py")):
        with open(os.path.join(d, "__init__.py"), "w") as f:
            pass

# Import PBIP utilities
ROOT_FOLDER = r"D:\NghienCuu\NghienCuuPaper\Source_Code\WSSS-01\src\modified_externals\ViLa-PIP"
sys.path.insert(0, ROOT_FOLDER)

def load_dataset_from_csv(csv_path):
    data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)  # Skip header
        for row in reader:
            if len(row) >= 3:
                case_id, slide_id, label_str = row[0], row[1], row[2]
                # Take first 4 labels for 4 classes
                labels = [int(x) for x in label_str.split(',')[:4]]
                # Derive fg/bg label (1 if any foreground class is 1, 0 otherwise)
                fg_bg_label = 1.0 if any(l > 0 for l in labels) else 0.0
                zip_path = r"D:\NghienCuu\NghienCuuPaper\Source_Code\data\data_BCSS-WSSS\BCSS-WSSS.zip"
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    img_paths = [f.filename for f in zip_ref.infolist() if f.filename.startswith(
                        'BCSS-WSSS/') and f.filename.endswith('.png')]
                    if any(slide_id in path for path in img_paths):
                        data.append((slide_id, labels, fg_bg_label))
                    else:
                        print(
                            f"Warning: slide_id {slide_id} not found in BCSS-WSSS.zip, skipping")
    return data[:100]  # Limit to 100 samples

try:
    process_list_path = r"D:\NghienCuu\NghienCuuPaper\Source_Code\WSSS-01\src\modified_externals\ViLa-PIP\process_list\process_list.csv"
    dataset_info = load_dataset_from_csv(process_list_path)

    class BCSSTrainingDataset:
        def __init__(self, transform=None, feature_paths=None, high_res_dir=None, high_res_transform=None):
            self.transform = transform
            self.high_res_transform = high_res_transform
            self.data = dataset_info[:70]  # 70 samples for training
            self.base_dir = r"D:\NghienCuu\NghienCuuPaper\Source_Code\data\data_BCSS-WSSS\BCSS-WSSS.zip"
            self.high_res_dir = high_res_dir or r"D:\NghienCuu\NghienCuuPaper\Source_Code\WSSS-01\src\modified_externals\ViLa-PIP\high_scale_dataset"
            self.feature_paths = feature_paths or []
            with zipfile.ZipFile(self.base_dir, 'r') as zip_ref:
                self.img_paths = [f for f in zip_ref.infolist() if f.filename.startswith(
                    'BCSS-WSSS/training/') and f.filename.endswith('.png')]

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            slide_id, labels, fg_bg_label = self.data[idx]
            with zipfile.ZipFile(self.base_dir, 'r') as zip_ref:
                for img_info in self.img_paths:
                    if slide_id in img_info.filename:
                        with zip_ref.open(img_info) as img_file:
                            img_low = np.array(
                                Image.open(img_file).convert('RGB'))
                        break
                else:
                    print(
                        f"Warning: Low-res image not found for slide_id {slide_id}, returning zeros")
                    return slide_id, torch.zeros(3, 224, 224), torch.zeros(3, 256, 256), torch.zeros(1, 2), torch.zeros(1, 2), torch.FloatTensor(labels), torch.FloatTensor([fg_bg_label]), torch.zeros(224, 224).long()

            high_img_path = os.path.join(
                self.high_res_dir, f"{slide_id}\\0_0.png")
            if os.path.exists(high_img_path):
                img_high = np.array(Image.open(high_img_path).convert('RGB'))
            else:
                print(
                    f"Warning: High-res image not found for slide_id {slide_id} at {high_img_path}, using low-res fallback")
                img_high = img_low

            if self.transform:
                augmented_low = self.transform(image=img_low)
                img_low = augmented_low['image']
            if self.high_res_transform:
                augmented_high = self.high_res_transform(image=img_high)
                img_high = augmented_high['image']

            coords_s = torch.tensor([112, 112], dtype=torch.float)
            coords_l = torch.tensor([128, 128], dtype=torch.float)
            return slide_id, img_low, img_high, coords_s, coords_l, torch.FloatTensor(labels), torch.FloatTensor([fg_bg_label]), torch.zeros(224, 224).long()

    class BCSSTestDataset:
        def __init__(self, split="valid", transform=None, feature_paths=None, high_res_dir=None, high_res_transform=None):
            self.transform = transform
            self.high_res_transform = high_res_transform
            self.data = dataset_info[70:85] if split == "valid" else dataset_info[85:100]
            self.base_dir = r"D:\NghienCuu\NghienCuuPaper\Source_Code\data\data_BCSS-WSSS\BCSS-WSSS.zip"
            self.high_res_dir = high_res_dir or r"D:\NghienCuu\NghienCuuPaper\Source_Code\WSSS-01\src\modified_externals\ViLa-PIP\high_scale_dataset"
            self.feature_paths = feature_paths or []
            with zipfile.ZipFile(self.base_dir, 'r') as zip_ref:
                self.img_paths = [f for f in zip_ref.infolist() if f.filename.startswith(
                    'BCSS-WSSS/test/img/') and f.filename.endswith('.png')]

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            slide_id, labels, fg_bg_label = self.data[idx]
            with zipfile.ZipFile(self.base_dir, 'r') as zip_ref:
                for img_info in self.img_paths:
                    if slide_id in img_info.filename:
                        with zip_ref.open(img_info) as img_file:
                            img_low = np.array(
                                Image.open(img_file).convert('RGB'))
                        break
                else:
                    print(
                        f"Warning: Low-res image not found for slide_id {slide_id}, returning zeros")
                    return slide_id, torch.zeros(3, 224, 224), torch.zeros(3, 256, 256), torch.zeros(1, 2), torch.zeros(1, 2), torch.FloatTensor(labels), torch.FloatTensor([fg_bg_label]), torch.zeros(224, 224).long()

            high_img_path = os.path.join(
                self.high_res_dir, f"{slide_id}\\0_0.png")
            if os.path.exists(high_img_path):
                img_high = np.array(Image.open(high_img_path).convert('RGB'))
            else:
                print(
                    f"Warning: High-res image not found for slide_id {slide_id} at {high_img_path}, using low-res fallback")
                img_high = img_low

            if self.transform:
                augmented_low = self.transform(image=img_low)
                img_low = augmented_low['image']
            if self.high_res_transform:
                augmented_high = self.high_res_transform(image=img_high)
                img_high = augmented_high['image']

            coords_s = torch.tensor([112, 112], dtype=torch.float)
            coords_l = torch.tensor([128, 128], dtype=torch.float)
            return slide_id, img_low, img_high, coords_s, coords_l, torch.FloatTensor(labels), torch.FloatTensor([fg_bg_label]), torch.zeros(224, 224).long()

    class BCSSWSSSDataset:
        def __init__(self, mask_name="val/mask", transform=None, feature_paths=None, high_res_dir=None, high_res_transform=None):
            self.transform = transform
            self.high_res_transform = high_res_transform
            self.data = dataset_info[70:85]
            self.base_dir = r"D:\NghienCuu\NghienCuuPaper\Source_Code\data\data_BCSS-WSSS\BCSS-WSSS.zip"
            self.high_res_dir = high_res_dir or r"D:\NghienCuu\NghienCuuPaper\Source_Code\WSSS-01\src\modified_externals\ViLa-PIP\high_scale_dataset"
            self.feature_paths = feature_paths or []
            with zipfile.ZipFile(self.base_dir, 'r') as zip_ref:
                self.img_paths = [f for f in zip_ref.infolist() if f.filename.startswith(
                    'BCSS-WSSS/val/img/') and f.filename.endswith('.png')]
                self.mask_paths = [f for f in zip_ref.infolist() if f.filename.startswith(
                    f'BCSS-WSSS/{mask_name}/') and f.filename.endswith('.png')]

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            slide_id, labels, fg_bg_label = self.data[idx]
            with zipfile.ZipFile(self.base_dir, 'r') as zip_ref:
                for img_info in self.img_paths:
                    if slide_id in img_info.filename:
                        with zip_ref.open(img_info) as img_file:
                            img_low = np.array(
                                Image.open(img_file).convert('RGB'))
                        break
                else:
                    print(
                        f"Warning: Low-res image not found for slide_id {slide_id}, returning zeros")
                    return slide_id, torch.zeros(3, 224, 224), torch.zeros(3, 256, 256), torch.zeros(1, 2), torch.zeros(1, 2), torch.FloatTensor(labels), torch.FloatTensor([fg_bg_label]), torch.zeros(224, 224).long()

            high_img_path = os.path.join(
                self.high_res_dir, f"{slide_id}\\0_0.png")
            if os.path.exists(high_img_path):
                img_high = np.array(Image.open(high_img_path).convert('RGB'))
            else:
                print(
                    f"Warning: High-res image not found for slide_id {slide_id} at {high_img_path}, using low-res fallback")
                img_high = img_low

            mask_info = next(
                (m for m in self.mask_paths if slide_id in m.filename), None)
            if mask_info:
                with zipfile.ZipFile(self.base_dir, 'r') as zip_ref:
                    with zip_ref.open(mask_info) as mask_file:
                        mask = np.array(Image.open(mask_file))
                        mask = torch.from_numpy(mask).long()
            else:
                mask = torch.zeros(224, 224).long()

            if self.transform:
                augmented_low = self.transform(image=img_low)
                img_low = augmented_low['image']
            if self.high_res_transform:
                augmented_high = self.high_res_transform(image=img_high)
                img_high = augmented_high['image']

            coords_s = torch.tensor([112, 112], dtype=torch.float)
            coords_l = torch.tensor([128, 128], dtype=torch.float)
            return slide_id, img_low, img_high, coords_s, coords_l, torch.FloatTensor(labels), torch.FloatTensor([fg_bg_label]), mask

except ImportError as e:
    raise ImportError(f"Failed to import or process BCSS datasets: {str(e)}")

global start_time, iters_per_epoch
start_time = datetime.datetime.now()

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default=None)
parser.add_argument("--gpu", type=int, default=0, help="gpu id")
args = parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def cal_eta(time0, cur_iter, total_iter):
    time_now = datetime.datetime.now()
    time_now = time_now.replace(microsecond=0)
    scale = (total_iter - cur_iter) / float(cur_iter) if cur_iter > 0 else 1
    delta = (time_now - time0)
    eta = (delta * scale)
    time_fin = time_now + eta
    eta = time_fin.replace(microsecond=0) - time_now
    return str(delta), str(eta)

def find_latest_checkpoint(ckpt_dir):
    checkpoint_files = [f for f in os.listdir(ckpt_dir) if f.startswith(
        "checkpoint_epoch_") and f.endswith(".pth")]
    if not checkpoint_files:
        return None, 0, 0
    epochs = [int(f.split("_epoch_")[1].split("_")[0] if "_stage2" not in f else f.split(
        "_epoch_")[1].split(".")[0]) for f in checkpoint_files]
    latest_epoch = max(epochs)
    latest_checkpoint = os.path.join(ckpt_dir, f"checkpoint_epoch_{latest_epoch}.pth" if latest_epoch in [int(f.split("_epoch_")[
                                     1].split(".")[0]) for f in checkpoint_files if "_stage2" not in f] else f"checkpoint_epoch_{latest_epoch}_stage2.pth")
    return latest_checkpoint, latest_epoch, (latest_epoch * iters_per_epoch)

def validate(model, data_loader, cfg, cls_loss_func, miou_metric=None):
    model.eval()
    loss_meter = AverageMeter()
    acc_all_meter = AverageMeter()
    acc_avg_meter = AverageMeter()
    miou_scores = []
    device = next(model.parameters()).device
    data_loader_list = list(DataLoader(data_loader.dataset, batch_size=1, shuffle=False, num_workers=0,
                            pin_memory=False, prefetch_factor=None, persistent_workers=False))[:min(100, len(data_loader.dataset))]
    data_loader = iter(data_loader_list)
    with torch.no_grad():
        for img_name, x_s, x_l, coord_s, coord_l, cls_labels, fg_bg_label, gt_label in tqdm(data_loader, desc="Validating", total=min(100, len(data_loader_list))):
            x_s = x_s.to(device).float()
            x_l = F.interpolate(x_l.to(device).float(), size=(
                224, 224), mode='bilinear', align_corners=True)
            cls_labels = cls_labels.to(device).float()
            fg_bg_label = fg_bg_label.to(device).float()
            print(
                f"Validation batch - cls_names: {cfg.model.text_prompt}, cls_labels shape: {cls_labels.shape}, fg_bg_label shape: {fg_bg_label.shape}, gt_label shape: {gt_label.shape}")
            try:
                outputs = model(x_s, fg_bg_labels=fg_bg_label)
                if len(outputs) == 8:
                    cls1, cam1, cls2, cam2, cls3, cam3, cls4, cam4 = outputs
                else:
                    cls1, cam1, cls2, cam2, cls3, cam3, cls4, cam4, mi_score, mil_logits, fg_bg_logits = outputs
                cls_outputs = cls4
                cams = cam4
                print(
                    f"Validation - cls4 shape: {cls4.shape}, cam4 shape: {cam4.shape}, fg_bg_logits shape: {fg_bg_logits.shape}")
            except ValueError as e:
                print(f"Validation error: {str(e)}")
                outputs = model(x_s, fg_bg_labels=fg_bg_label)
                cls1, cam1, cls2, cam2, cls3, cam3, cls4, cam4 = outputs[:8]
                cls_outputs = cls4
                cams = cam4
            cls_merge = merge_to_parent_predictions(
                cls_outputs, None, method=cfg.train.merge_train)
            loss = cls_loss_func(cls_merge, cls_labels)  # Only for 4 classes
            loss_meter.update(loss.item(), x_s.size(0))
            cls_pred = torch.argmax(torch.softmax(cls_merge, dim=1), dim=1).float(
            ).unsqueeze(1)  # Convert to one-hot for comparison
            acc_all = (cls_pred == cls_labels.argmax(dim=1).unsqueeze(1)).all(
                dim=1).float().mean().item() * 100
            acc_avg = ((cls_pred == cls_labels.argmax(dim=1).unsqueeze(
                1)).float().mean(dim=0)).mean().item() * 100
            acc_all_meter.update(acc_all, x_s.size(0))
            acc_avg_meter.update(acc_avg, x_s.size(0))
            if miou_metric and gt_label is not None:
                cams = F.interpolate(
                    cams, size=gt_label.shape[-2:], mode='bilinear', align_corners=False)
                # Combine 4-class CAM with fg/bg for 5-class prediction
                fg_bg_mask = (torch.sigmoid(fg_bg_logits) > 0.5).float(
                ).unsqueeze(1).expand(-1, 1, *cams.shape[2:])
                # [batch_size, 5, H, W]
                cams_5 = torch.cat((cams[:, :4, :, :], fg_bg_mask), dim=1)
                preds = torch.argmax(cams_5, dim=1)
                miou = miou_metric(preds, gt_label.to(device)).item()
                miou_scores.append(miou)
    model.train()
    # Return 5 scores for 4 classes + bg
    return acc_all_meter.avg, acc_avg_meter.avg, np.array(miou_scores) if miou_scores else np.zeros(5), loss_meter.avg

def generate_cam(model, data_loader, cfg, is_final=False, epoch=None, pseudo_labels=False):
    model.eval()
    device = next(model.parameters()).device
    pred_dir = cfg.work_dir.pred_dir
    result_dir = r"D:\NghienCuu\NghienCuuPaper\Source_Code\WSSS-01\result"
    input_size = cfg.dataset.input_size
    if epoch is not None:
        pred_dir = os.path.join(pred_dir, f"epoch_{epoch}")
        if pseudo_labels:
            pred_dir = os.path.join(pred_dir, "pseudo_labels")
        result_dir = os.path.join(result_dir, f"epoch_{epoch}")
        os.makedirs(pred_dir, exist_ok=True)
        os.makedirs(result_dir, exist_ok=True)
    data_loader_list = list(DataLoader(data_loader.dataset, batch_size=1, shuffle=False, num_workers=0,
                            pin_memory=False, prefetch_factor=None, persistent_workers=False))[:min(100, len(data_loader.dataset))]
    data_loader = iter(data_loader_list)
    total_images = min(100, len(data_loader_list))
    with torch.no_grad():
        for img_name, x_s, x_l, coord_s, coord_l, cls_labels, fg_bg_label, _ in tqdm(data_loader, desc=f"Generating Pseudo-Labels{' (Final)' if is_final else f' Epoch {epoch}'}: {total_images} images", total=total_images):
            try:
                x_s = x_s.to(device).float()
                x_l = F.interpolate(x_l.to(device).float(), size=(
                    224, 224), mode='bilinear', align_corners=True)
                coord_s = coord_s.to(device).float()
                coord_l = coord_l.to(device).float()
                fg_bg_label = fg_bg_label.to(device).float()
                cam_low, cam_high, _ = model.generate_cam(
                    x_s, coord_s, x_l, coord_l, None)
                for name, cam in zip(img_name, cam_high):
                    cam = F.interpolate(cam.unsqueeze(0), size=(
                        input_size[0], input_size[1]), mode='bilinear', align_corners=False)
                    save_path = os.path.join(pred_dir, f"{name}_cam.pth")
                    torch.save(cam.cpu(), save_path)
                    # Generate binary mask with 5 classes (4 + bg)
                    binary_mask = torch.zeros_like(cam[:, 0:1, :, :])
                    for i in range(4):  # 4 foreground classes
                        binary_mask[cam[:, i:i+1, :, :] >
                                    0.5] = i + 1  # 1-4 for classes
                    # 5 for background
                    binary_mask[torch.sigmoid(cam[:, 4:5, :, :]) > 0.5] = 5
                    mask_save_path = os.path.join(pred_dir, f"{name}_mask.pth")
                    torch.save(binary_mask.cpu(), mask_save_path)
                    print(f"Saved CAM: {save_path}, Mask: {mask_save_path}")
            except Exception as e:
                print(f"Error generating CAM for {img_name}: {str(e)}")
                continue
    model.train()

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

def train(cfg):
    global iters_per_epoch, start_time
    print("\nInitializing training...")
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    set_seed(42)

    print("Full config:", OmegaConf.to_yaml(cfg))

    time0 = datetime.datetime.now().replace(microsecond=0)
    dataset_config = cfg['dataset']
    model_config = cfg['model']
    train_config = cfg['train']
    work_dir = cfg['work_dir']
    optimizer_config = cfg['optimizer']
    scheduler_config = cfg['scheduler']

    model_config.text_prompt = [
        "A WSI of Tumor at low resolution with visually descriptive characteristics of irregularly shaped regions, dense cellularity, heterogeneous staining, and distortion of adjacent structures due to growth.",
        "A WSI of Tumor at high resolution with visually descriptive characteristics of atypical cells, enlarged nuclei, prominent nucleoli, high nuclear-to-cytoplasmic ratio, mitotic figures, and invasive patterns.",
        "A WSI of Stroma at low resolution with visually descriptive characteristics of fibrous connective tissue, lighter staining, low cellular density, and surrounding or infiltrating tumor areas.",
        "A WSI of Stroma at high resolution with visually descriptive characteristics of elongated fibroblasts, collagen bundles, eosinophilic matrix, blood vessels, and occasional inflammatory cells.",
        "A WSI of Lymphocyte at low resolution with visually descriptive characteristics of small dark clusters or infiltrates, often at tumor-stroma interfaces, appearing as speckled blue-purple areas.",
        "A WSI of Lymphocyte at high resolution with visually descriptive characteristics of small round cells, hyperchromatic nuclei, scant cytoplasm, and clustering in immune responses.",
        "A WSI of Necrosis at low resolution with visually descriptive characteristics of pale amorphous zones, loss of structure, hypoeosinophilic appearance, and contrast with viable tissue.",
        "A WSI of Necrosis at high resolution with visually descriptive characteristics of cellular debris, karyorrhectic nuclei, cytoplasmic remnants, and infiltration by inflammatory cells.",
        "A WSI of Background at low resolution with normal tissue structure, uniform staining, and absence of pathological features.",
        "A WSI of Background at high resolution with normal cells, regular nuclei, balanced nuclear-to-cytoplasmic ratio, and no signs of malignancy."
    ]
    model_config.input_size = tuple(model_config.input_size)
    dataset_config.input_size = tuple(dataset_config.input_size)
    model_config.label_feature_path = list(
        model_config.label_feature_path) if model_config.label_feature_path is not None else []

    ckpt_dir = work_dir['ckpt_dir']
    pred_dir = work_dir['pred_dir']
    csv_dir = work_dir['csv_dir']
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    csv_file = os.path.join(
        csv_dir, f"training_log_{start_time.strftime('%Y%m%d_%H%M%S')}.csv")
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Iteration", "Loss",
                        "Accuracy", "Validation_mIoU", "Best_mIoU"])

    train_transform = A.Compose([
        A.Resize(height=224, width=224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    val_transform = A.Compose([
        A.Resize(height=224, width=224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    high_res_transform = A.Compose([
        A.Resize(height=256, width=256),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    print("Preprocessing training dataset...")
    train_dataset = BCSSTrainingDataset(
        transform=train_transform, feature_paths=model_config.label_feature_path, high_res_transform=high_res_transform)
    train_dataset = Subset(train_dataset, range(min(100, len(train_dataset))))
    print(f"Training dataset size: {len(train_dataset)}")

    print("Preprocessing validation dataset...")
    val_dataset = BCSSTestDataset(split="valid", transform=val_transform,
                                  feature_paths=model_config.label_feature_path, high_res_transform=high_res_transform)
    val_dataset = Subset(val_dataset, range(min(100, len(val_dataset))))
    print(f"Validation dataset size: {len(val_dataset)}")

    print("Preprocessing test dataset...")
    test_dataset = BCSSTestDataset(split="test", transform=val_transform,
                                   feature_paths=model_config.label_feature_path, high_res_transform=high_res_transform)
    test_dataset = Subset(test_dataset, range(min(100, len(test_dataset))))
    print(f"Test dataset size: {len(test_dataset)}")

    num_workers = 0
    train_loader = DataLoader(train_dataset, batch_size=cfg.train.samples_per_gpu, shuffle=True, num_workers=num_workers,
                              pin_memory=False, prefetch_factor=None, persistent_workers=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=num_workers,
                            pin_memory=False, prefetch_factor=None, persistent_workers=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers,
                             pin_memory=False, prefetch_factor=None, persistent_workers=False)

    try:
        print("Initializing ViLa_MIL_Model...")
        vilamil_model = ViLa_MIL_Model(
            config=model_config, num_classes=4)  # 4 classes
        print(
            f"ViLa_MIL_Model initialized with hidden_size: {vilamil_model.D}")
        vilamil_model = vilamil_model.to(device)
        print("ViLa_MIL_Model moved to device:", device)

        print("Initializing ClsNetwork...")
        model = ClsNetwork(
            backbone=vilamil_model,
            stride=model_config['backbone'].get('stride', [4, 2, 2, 1]),
            cls_num_classes=4,  # 4 classes
            n_ratio=model_config['n_ratio'],
            pretrained=train_config['pretrained'],
            l_fea_path=model_config['label_feature_path'],
            text_prompt=model_config.text_prompt
        )
        print("ClsNetwork initialized successfully")
        model = model.to(device)
        print("ClsNetwork moved to device:", device)

        print(
            f"Successfully loaded ClsNetwork with ViLa_MIL_Model backbone and num_classes: {model.cls_num_classes}")
    except Exception as e:
        print(f"Error during model initialization: {str(e)}")
        raise RuntimeError(
            f"Failed to initialize ClsNetwork with ViLa_MIL_Model: {str(e)}")
    model = model.train()

    iters_per_epoch = len(train_loader)
    max_iters = train_config['epoch'] * iters_per_epoch
    warmup_iter = scheduler_config['warmup_iter'] * iters_per_epoch
    scaler = torch.amp.GradScaler(enabled=False)

    optimizer = PolyWarmupAdamW(
        params=model.parameters(), lr=optimizer_config['learning_rate'], weight_decay=optimizer_config['weight_decay'],
        betas=optimizer_config['betas'], warmup_iter=warmup_iter, max_iter=max_iters,
        warmup_ratio=scheduler_config['warmup_ratio'], power=scheduler_config['power']
    )

    start_epoch, start_iter, best_miou = 0, 0, 0.0
    latest_checkpoint, latest_epoch, start_iter = find_latest_checkpoint(
        ckpt_dir)
    if latest_checkpoint:
        print(f"Loading checkpoint from: {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        state_dict = checkpoint["model"]
        new_state_dict = {}
        for k, v in state_dict.items():
            if 'classifier' in k and v.shape[0] != 4:  # Adjust for 4 classes
                print(
                    f"Adjusting classifier weights from {v.shape[0]} to 4 classes")
                new_state_dict[k] = v[:4]
            elif 'l_fc4.conv.weight' in k and v.shape[0] != 4:
                new_state_dict[k] = v[:4]
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=False)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = latest_epoch
        best_miou = checkpoint.get("best_miou", 0.0)
        print(f"Resumed from epoch {start_epoch + 1}, iteration {start_iter}")

    loss_function = nn.CrossEntropyLoss().to(device)  # For 4 classes
    fg_bg_loss = nn.BCEWithLogitsLoss().to(device)  # For fg/bg
    mask_adapter = MaskAdapter_DynamicThreshold(
        alpha=train_config['mask_adapter_alpha'])
    feature_extractor = FeatureExtractor(
        mask_adapter=mask_adapter,
        high_path=model_config.label_feature_path[0] if model_config.label_feature_path else None,
        low_path=model_config.label_feature_path[1] if model_config.label_feature_path else None
    )
    fg_loss_fn = InfoNCELossFG(temperature=0.07).to(device)
    bg_loss_fn = InfoNCELossBG(temperature=0.07).to(device)
    miou_metric = JaccardIndex(task="multiclass", num_classes=5).to(
        device)  # 5 for 4 classes + bg
    text_projector = nn.Linear(1024, 512).to(device)

    print("\nStarting Stage 1 Training (Classification & CAM Generation)...")
    for epoch in range(start_epoch, train_config['epoch']):
        model.train()
        for n_iter, (img_name, x_s, x_l, coord_s, coord_l, cls_labels, fg_bg_label, gt_label) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{train_config['epoch']}", total=len(train_loader))):
            if epoch == start_epoch and n_iter < start_iter:
                continue
            print(
                f"Training batch - cls_names: {cfg.model.text_prompt}, cls_labels shape: {cls_labels.shape}, fg_bg_label shape: {fg_bg_label.shape}, gt_label shape: {gt_label.shape}")
            print(
                f"cls_labels max: {cls_labels.max().item()}, min: {cls_labels.min().item()}")
            try:
                x_s = x_s.to(device).float()
                x_l = x_l.to(device).float()
                coord_s = coord_s.to(device).float()
                coord_l = coord_l.to(device).float()
                cls_labels = cls_labels.to(
                    device).float()  # One-hot for 4 classes
                fg_bg_label = fg_bg_label.to(device).float()

                with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    text_prompts = model.backbone.prompt_learner()
                    tokenized_prompts = model.backbone.prompt_learner.tokenized_prompts
                    text_features = model.backbone.text_encoder(
                        text_prompts, tokenized_prompts)
                    text_features = text_projector(text_features)
                    print(
                        f"text_features shape after projection: {text_features.shape}")
                    outputs = model(x_s, fg_bg_labels=fg_bg_label)
                    cls1, cam1, cls2, cam2, cls3, cam3, cls4, cam4, mi_score, mil_logits, fg_bg_logits = outputs

                    cls1_merge = merge_to_parent_predictions(
                        cls1, None, method=cfg.train.merge_train)
                    cls2_merge = merge_to_parent_predictions(
                        cls2, None, method=cfg.train.merge_train)
                    cls3_merge = merge_to_parent_predictions(
                        cls3, None, method=cfg.train.merge_train)
                    cls4_merge = merge_to_parent_predictions(
                        cls4, None, method=cfg.train.merge_train)

                    # Probabilities for 4 classes
                    cls4 = F.softmax(cls4, dim=1)
                    subclass_labels = cls_labels  # Use cls_labels as one-hot for 4 classes
                    cls4_bir = cls4 * subclass_labels
                    print(
                        f"cls4 shape: {cls4.shape}, max: {cls4.max().item()}, min: {cls4.min().item()}")
                    print(
                        f"subclass_labels shape: {subclass_labels.shape}, max: {subclass_labels.max().item()}, min: {subclass_labels.min().item()}")
                    print(
                        f"cls4_bir max: {cls4_bir.max().item()}, min: {cls4_bir.min().item()}")

                    batch_info = feature_extractor.process_batch(
                        img_name, cam4, cls4_bir, csv_dir, epoch, n_iter, device, x_s, x_l)
                    if batch_info is None:
                        print(
                            f"Warning: Feature extraction failed for batch {img_name}, using default features")
                        with torch.no_grad():
                            fg_features = model.projector.ImageMLP(
                                x_s).mean(dim=[2, 3])
                            bg_features = model.projector.ImageMLP(
                                x_l).mean(dim=[2, 3])
                    else:
                        fg_features = batch_info['fg_features']
                        bg_features = batch_info['bg_features']
                        print(
                            f"batch_info fg_features shape: {fg_features.shape}, bg_features shape: {bg_features.shape}")

                    set_info = pair_features(
                        fg_features, bg_features, text_features, cls4_bir)
                    fg_features, bg_features, fg_pro, bg_pro = set_info['fg_features'], set_info[
                        'bg_features'], set_info['fg_text'], set_info['bg_text']
                    print(
                        f"paired fg_features shape: {fg_features.shape}, bg_features shape: {bg_features.shape}")

                    if fg_features.numel() == 0 or bg_features.numel() == 0:
                        print("Skipping loss computation due to empty feature sets")
                        continue

                    print(
                        f"fg_loss_fn - fg_features shape: {fg_features.shape}, fg_pro shape: {fg_pro.shape}, bg_pro shape: {bg_pro.shape}")
                    fg_loss = fg_loss_fn(fg_features, fg_pro, bg_pro)
                    bg_loss = bg_loss_fn(bg_features, fg_pro, bg_pro)
                    loss_sim = fg_loss + bg_loss

                    loss1 = loss_function(cls1_merge, cls_labels.argmax(dim=1))
                    loss2 = loss_function(cls2_merge, cls_labels.argmax(dim=1))
                    loss3 = loss_function(cls3_merge, cls_labels.argmax(dim=1))
                    loss4 = loss_function(cls4_merge, cls_labels.argmax(dim=1))
                    fg_bg_loss_value = fg_bg_loss(fg_bg_logits, fg_bg_label)
                    cls_loss = cfg.train.l1 * loss1 + cfg.train.l2 * \
                        loss2 + cfg.train.l3 * loss3 + cfg.train.l4 * loss4
                    loss = cls_loss + (loss_sim + 0.0005 * torch.mean(cam4)) * \
                        cfg.train.l5 + fg_bg_loss_value * 0.1  # Weight fg/bg loss

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                delta, eta = cal_eta(time0, n_iter + 1, max_iters)
                cur_lr = optimizer.param_groups[0]['lr']
                cls_pred4 = torch.argmax(torch.softmax(
                    cls4_merge, dim=1), dim=1).float().unsqueeze(1)
                all_cls_acc4 = (cls_pred4 == cls_labels.argmax(
                    dim=1).unsqueeze(1)).all(dim=1).float().mean() * 100
                avg_cls_acc4 = ((cls_pred4 == cls_labels.argmax(
                    dim=1).unsqueeze(1)).float().mean(dim=0)).mean() * 100
                print(f"Epoch: {epoch + 1}/{train_config['epoch']}; Iter: {n_iter + 1}/{max_iters}; "
                      f"Elapsed: {delta}; ETA: {eta}; LR: {cur_lr:.3e}; Loss: {loss.item():.4f}; "
                      f"Acc4: {all_cls_acc4:.2f}/{avg_cls_acc4:.2f}")
                with open(csv_file, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [epoch + 1, n_iter + 1, loss.item(), all_cls_acc4.item(), "", best_miou])

                try:
                    checkpoint_path = os.path.join(
                        ckpt_dir, f"checkpoint_epoch_{epoch + 1}_iter_{n_iter + 1}_stage1.pth")
                    torch.save({"cfg": cfg, "epoch": epoch + 1, "iter": n_iter, "model": model.state_dict(),
                               "optimizer": optimizer.state_dict(), "best_miou": best_miou},
                               checkpoint_path, _use_new_zipfile_serialization=True)
                    print(
                        f"Saved checkpoint at iteration {n_iter + 1}: {checkpoint_path}")
                except Exception as e:
                    print(
                        f"Error saving checkpoint at iteration {n_iter + 1}: {str(e)}")

                if (n_iter + 1) % 2 == 0:
                    print(
                        f"Generating pseudo-labels for iteration {n_iter + 1}...")
                    generate_cam(model, train_loader, cfg, epoch=epoch + 1)

                if (n_iter + 1) % iters_per_epoch == 0 or (n_iter + 1) == max_iters:
                    val_all_acc4, val_avg_acc4, fuse234_score, val_cls_loss = validate(
                        model=model, data_loader=val_loader, cfg=cfg, cls_loss_func=loss_function, miou_metric=miou_metric
                    )
                    miou = fuse234_score.mean() if len(fuse234_score) > 0 else 0.0
                    print(
                        f"Epoch {epoch + 1}/{train_config['epoch']}, Validation mIoU: {miou:.4f}")
                    with open(csv_file, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(
                            [epoch + 1, n_iter + 1, val_cls_loss, val_all_acc4, miou, best_miou])

                    if miou > best_miou:
                        best_miou = miou
                        save_path = os.path.join(
                            ckpt_dir, f"best_stage1_epoch_{epoch + 1}.pth")
                        torch.save({"cfg": cfg, "epoch": epoch, "iter": n_iter, "model": model.state_dict(),
                                   "optimizer": optimizer.state_dict(), "best_miou": best_miou},
                                   save_path, _use_new_zipfile_serialization=True)
                        print(
                            f"Saved best model with mIoU: {best_miou:.4f} at {save_path}")

                    checkpoint_path = os.path.join(
                        ckpt_dir, f"checkpoint_epoch_{epoch + 1}_stage1.pth")
                    torch.save({"cfg": cfg, "epoch": epoch + 1, "iter": n_iter, "model": model.state_dict(),
                               "optimizer": optimizer.state_dict(), "best_miou": best_miou},
                               checkpoint_path, _use_new_zipfile_serialization=True)
                    print(
                        f"Saved checkpoint Stage 1 at epoch {epoch + 1}: {checkpoint_path}")

                    print(f"Generating pseudo-labels for epoch {epoch + 1}...")
                    generate_cam(model, train_loader, cfg, epoch=epoch + 1)

            except Exception as e:
                print(f"Error in training iteration {n_iter + 1}: {str(e)}")
                continue

    print("\nGenerating CAMs for pseudo-labeling...")
    generate_cam(model, train_loader, cfg, epoch="pseudo_labeling")

    print("\nStarting Stage 2 Training (Refinement with Pseudo-Labels)...")
    pseudo_dataset = BCSSWSSSDataset(mask_name="val/mask", transform=val_transform,
                                     feature_paths=model_config.label_feature_path, high_res_transform=high_res_transform)
    pseudo_dataset = Subset(pseudo_dataset, range(
        min(100, len(pseudo_dataset))))
    print(f"Pseudo dataset size: {len(pseudo_dataset)}")
    pseudo_loader = DataLoader(pseudo_dataset, batch_size=cfg.train.samples_per_gpu, shuffle=True, num_workers=num_workers,
                               pin_memory=False, prefetch_factor=None, persistent_workers=False)

    seg_criterion = nn.CrossEntropyLoss().to(device)  # For 5 classes (4 + bg)
    for epoch in range(start_epoch, train_config['epoch']):
        model.train()
        for n_iter, (img_name, x_s, x_l, coord_s, coord_l, cls_labels, fg_bg_label, masks) in enumerate(tqdm(pseudo_loader, desc=f"Training Stage 2 Epoch {epoch+1}/{train_config['epoch']}", total=len(pseudo_loader))):
            try:
                x_s = x_s.to(device).float()
                x_l = F.interpolate(x_l.to(device).float(), size=(
                    224, 224), mode='bilinear', align_corners=True)
                masks = masks.to(device).long()
                with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    outputs = model(x_s, fg_bg_labels=fg_bg_label)
                    cams = outputs[7] if len(
                        outputs) >= 8 else outputs[3]  # Use cam4
                    seg_outputs = F.interpolate(
                        cams, size=masks.shape[-2:], mode='bilinear', align_corners=False)
                    seg_loss = seg_criterion(seg_outputs, masks)

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(seg_loss).backward()
                scaler.step(optimizer)
                scaler.update()

                n_iter_total = epoch * iters_per_epoch + n_iter
                with open(csv_file, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [epoch + 1, n_iter_total, seg_loss.item(), "", "", best_miou])

                try:
                    checkpoint_path = os.path.join(
                        ckpt_dir, f"checkpoint_epoch_{epoch + 1}_iter_{n_iter + 1}_stage2.pth")
                    torch.save({"cfg": cfg, "epoch": epoch + 1, "iter": n_iter, "model": model.state_dict(),
                               "optimizer": optimizer.state_dict(), "best_miou": best_miou},
                               checkpoint_path, _use_new_zipfile_serialization=True)
                    print(
                        f"Saved checkpoint at iteration {n_iter + 1}: {checkpoint_path}")
                except Exception as e:
                    print(
                        f"Error saving checkpoint at iteration {n_iter + 1}: {str(e)}")
            except Exception as e:
                print(f"Error in Stage 2 iteration {n_iter + 1}: {str(e)}")
                continue

        miou = validate(model, val_loader, cfg,
                        loss_function, miou_metric)[2].mean()
        print(
            f"Epoch {epoch + 1}/{train_config['epoch']}, Validation mIoU: {miou:.4f}")
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(
                [epoch + 1, n_iter_total, seg_loss.item(), "", miou, best_miou])

        if miou > best_miou:
            best_miou = miou
            save_path = os.path.join(
                ckpt_dir, f"best_stage2_epoch_{epoch + 1}.pth")
            torch.save({"cfg": cfg, "epoch": epoch, "iter": n_iter_total, "model": model.state_dict(),
                       "optimizer": optimizer.state_dict(), "best_miou": best_miou},
                       save_path, _use_new_zipfile_serialization=True)
            print(
                f"Saved best model with mIoU: {best_miou:.4f} at {save_path}")

        checkpoint_path = os.path.join(
            ckpt_dir, f"checkpoint_epoch_{epoch + 1}_stage2.pth")
        torch.save({"cfg": cfg, "epoch": epoch + 1, "iter": n_iter_total, "model": model.state_dict(),
                   "optimizer": optimizer.state_dict(), "best_miou": best_miou},
                   checkpoint_path, _use_new_zipfile_serialization=True)
        print(
            f"Saved checkpoint Stage 2 at epoch {epoch + 1}: {checkpoint_path}")

        print(f"Generating pseudo-labels for epoch {epoch + 1}...")
        generate_cam(model, train_loader, cfg, epoch=epoch + 1)

    print("\nFinal Evaluation...")
    test_all_acc4, test_avg_acc4, fuse234_score, test_cls_loss = validate(
        model=model, data_loader=test_loader, cfg=cfg, cls_loss_func=loss_function, miou_metric=miou_metric
    )
    miou = fuse234_score.mean() if len(fuse234_score) > 0 else 0.0
    print(
        f"Test mIoU: {miou:.4f}, Test all acc4: {test_all_acc4:.6f}, Test avg acc4: {test_avg_acc4:.6f}")
    print(f"Fuse234 score: {fuse234_score}")
    for i, score in enumerate(fuse234_score):
        print(f"Class {i}: {score:.6f}")

    print("\nGenerating Final CAMs...")
    best_model_path = os.path.join(ckpt_dir, "best_stage2.pth")
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        print(f"Loaded best model from: {best_model_path}")
    generate_cam(model, train_loader, cfg, is_final=True)

    try:
        final_checkpoint_path = os.path.join(ckpt_dir, "final_checkpoint.pth")
        torch.save({"cfg": cfg, "epoch": train_config['epoch'], "iter": n_iter_total, "model": model.state_dict(),
                   "optimizer": optimizer.state_dict(), "best_miou": best_miou},
                   final_checkpoint_path, _use_new_zipfile_serialization=True)
        print(f"Saved final checkpoint at: {final_checkpoint_path}")
    except Exception as e:
        print(f"Error saving final checkpoint: {str(e)}")

    end_time = datetime.datetime.now()
    print(f'Total training time: {end_time - start_time}')


if __name__ == "__main__":
    if not args.config:
        raise ValueError(
            "No config file specified. Please provide a --config argument.")
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found at: {args.config}")
    if not args.config.endswith(('.yaml', '.yml')):
        raise TypeError(
            f"Config file must be a YAML file (.yaml or .yml), got: {args.config}")
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            content = f.read()
            if not content.strip():
                raise ValueError(f"Config file is empty: {args.config}")
        cfg = OmegaConf.load(args.config)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load config file {args.config}: {str(e)}")

    cfg.dataset.cls_num_classes = 5
    cfg.model.n_ratio = 1.0
    cfg.model.backbone.stride = [4, 2, 2, 1]
    cfg.model.label_feature_path = [
        r"D:\NghienCuu\NghienCuuPaper\Source_Code\WSSS-01\src\modified_externals\ViLa-PIP\features_extraction\dataset_features_extraction\high",
        r"D:\NghienCuu\NghienCuuPaper\Source_Code\WSSS-01\src\modified_externals\ViLa-PIP\features_extraction\dataset_features_extraction\low"
    ]
    cfg.dataset.input_size = [224, 224]  # Low-res size for x_s
    cfg.model.input_size = [224, 224]    # Low-res size for x_s
    if cfg.model.n_ratio <= 0 or not isinstance(cfg.model.n_ratio, (int, float)):
        print(f"Warning: Invalid n_ratio {cfg.model.n_ratio}, setting to 1")
        cfg.model.n_ratio = 1
    timestamp = "{0:%Y-%m-%d-%H-%M}".format(datetime.datetime.now())
    cfg.work_dir.dir = os.path.dirname(args.config)
    cfg.work_dir.ckpt_dir = os.path.join(cfg.work_dir.ckpt_dir, timestamp)
    cfg.work_dir.pred_dir = os.path.join(cfg.work_dir.pred_dir, timestamp)
    cfg.work_dir.csv_dir = os.path.join(cfg.work_dir.csv_dir, timestamp)
    os.makedirs(cfg.work_dir.dir, exist_ok=True)
    os.makedirs(cfg.work_dir.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.pred_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.csv_dir, exist_ok=True)
    print('\nArgs: %s' % args)
    print('\nConfigs: %s' % cfg)
    train(cfg=cfg)
    