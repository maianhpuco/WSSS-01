import argparse
import csv
import datetime
import os
import sys
import numpy as np
import cv2 as cv
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
import matplotlib.pyplot as plt
import h5py
import albumentations as A
from albumentations.pytorch import ToTensorV2
from scipy.ndimage import binary_dilation
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from torch.utils.data import ConcatDataset, Subset

# Import required modules from model and utils
ROOT_FOLDER = r"D:\NghienCuu\NghienCuuPaper\Source_Code\WSSS-01\src\modified_externals\ViLa-PIP"
sys.path.insert(0, ROOT_FOLDER)

from model.model_ViLa_MIL import ViLa_MIL_Model
from model.projector import PLIPProjector
from model.model import ClsNetwork
from utils.pyutils import AverageMeter, set_seed
from utils.fgbg_feature import FeatureExtractor, MaskAdapter_DynamicThreshold
from utils.contrast_loss import InfoNCELossFG, InfoNCELossBG
from utils.hierarchical_utils import pair_features
from datasets.bcss_wsss import BCSS_WSSSTrainingDataset, BCSSWSSSDataset

# Define H5 directories
H5_TRAIN_DIR = r"D:\NghienCuu\NghienCuuPaper\Source_Code\WSSS-01\src\modified_externals\ViLa-PIP\features_extraction\dataset_features_extraction"
H5_TEST_DIR = r"D:\NghienCuu\NghienCuuPaper\Source_Code\WSSS-01\src\modified_externals\ViLa-PIP\features_extraction\test_feature_extraction"
H5_VAL_DIR = r"D:\NghienCuu\NghienCuuPaper\Source_Code\WSSS-01\src\modified_externals\ViLa-PIP\features_extraction\val_feature_extraction"

BASE_DIR = r"D:\NghienCuu\NghienCuuPaper\Source_Code\data\data_BCSS-WSSS\BCSS-WSSS"

# Updated palette to match BCSS dataset colors: Red, Green, Blue, Purple (4 classes)
palette = np.array([
    [255, 0, 0],    # TUM - Red
    [0, 255, 0],    # STR - Green
    [0, 0, 255],    # LYM - Blue
    [128, 0, 128],  # NEC - Purple
])

class_descriptions = {
    0: ("TUM", "Tumor", tuple(palette[0])),      # Red
    1: ("STR", "Stroma", tuple(palette[1])),     # Green
    2: ("LYM", "Lymphocyte", tuple(palette[2])), # Blue
    3: ("NEC", "Necrosis", tuple(palette[3])),   # Purple
}

try:
    from torchmetrics import JaccardIndex, Dice
    HAS_TORCHMETRICS = True
except ImportError:
    print("Warning: torchmetrics not found. Using custom implementation.")
    HAS_TORCHMETRICS = False

    def dice_score(pred, target, num_classes=4, ignore_index=None):
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        if ignore_index is not None:
            mask = target != ignore_index
            pred = pred[mask]
            target = target[mask]
        intersection = torch.zeros(num_classes).to(pred.device)
        union = torch.zeros(num_classes).to(pred.device)
        for class_idx in range(num_classes):
            pred_class = (pred == class_idx).float()
            target_class = (target == class_idx).float()
            intersection[class_idx] = (pred_class * target_class).sum()
            union[class_idx] = pred_class.sum() + target_class.sum()
        dice = (2. * intersection + 1e-8) / (union + 1e-8)
        return dice.mean()
    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.avg = self.sum

class DiceMetric(nn.Module):
    def __init__(self, num_classes=4, ignore_index=None, device=None):
        super(DiceMetric, self).__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.device = device
        self.preds = []
        self.targets = []

    def update(self, pred, target):
        pred = pred.to(self.device)
        target = target.to(self.device)
        self.preds.append(pred.cpu())
        self.targets.append(target.cpu())

    def compute(self):
        if not self.preds or not self.targets:
            return torch.tensor(0.0, device=self.device)
        preds = torch.cat(self.preds).view(-1).to(self.device)
        targets = torch.cat(self.targets).view(-1).to(self.device)
        score = dice_score(preds, targets, self.num_classes, self.ignore_index) * 100
        self.reset()
        return score

    def reset(self):
        self.preds = []
        self.targets = []

    def to(self, device):
        self.device = device
        return self

def frequency_weighted_iou(pred, target, num_classes=4, ignore_index=None):
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    if ignore_index is not None:
        mask = target != ignore_index
        pred = pred[mask]
        target = target[mask]

    freq = torch.histc(target.float(), bins=num_classes, min=0, max=num_classes-1) / target.numel()
    iou_per_class = torch.zeros(num_classes).to(pred.device)
    for class_idx in range(num_classes):
        pred_class = (pred == class_idx).float()
        target_class = (target == class_idx).float()
        intersection = (pred_class * target_class).sum()
        union = pred_class.sum() + target_class.sum() - intersection
        iou_per_class[class_idx] = intersection / (union + 1e-8) if union > 0 else 0
    fw_iou = torch.sum(freq * iou_per_class)
    return fw_iou

def boundary_iou(pred, target, num_classes=4, ignore_index=None, boundary_width=1):
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    if ignore_index is not None:
        mask = target != ignore_index
        pred = pred[mask]
        target = target[mask]

    def get_boundary(mask, width):
        dilated = binary_dilation(mask.cpu().numpy(), iterations=width).astype(np.uint8)
        eroded = binary_dilation(mask.cpu().numpy(), iterations=width-1).astype(np.uint8)
        boundary = dilated - eroded
        return torch.from_numpy(boundary).to(pred.device).float()

    iou_per_class = torch.zeros(num_classes).to(pred.device)
    for class_idx in range(num_classes):
        pred_class = (pred == class_idx).float()
        target_class = (target == class_idx).float()
        pred_boundary = get_boundary(pred_class.view_as(target).cpu(), boundary_width)
        target_boundary = get_boundary(target_class.view_as(target).cpu(), boundary_width)
        intersection = (pred_boundary * target_boundary).sum()
        union = pred_boundary.sum() + target_boundary.sum() - intersection
        iou_per_class[class_idx] = intersection / (union + 1e-8) if union > 0 else 0
    b_iou = iou_per_class.mean()
    return b_iou

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


def generate_cam(model, x, coords, cls_labels, fg_bg_labels, feature_extractor, loss_function, fg_loss_fn, bg_loss_fn, h5_path=None, img_name=None, save_dir=None, epoch=None):
    """
    Generate Class Activation Maps (CAMs) and binary masks for the given inputs.
    
    Args:
        model: The neural network model (e.g., ClsNetwork).
        x: Input images, shape [batch_size, channels, h, w].
        coords: Coordinate data, shape [batch_size, 2] or None.
        cls_labels: Class labels, shape [batch_size, num_classes] (one-hot or probabilities).
        fg_bg_labels: Foreground-background labels, shape [batch_size, 1].
        feature_extractor: Object to extract foreground and background features.
        loss_function: Loss function for classification (e.g., BCEWithLogitsLoss).
        fg_loss_fn: Loss function for foreground features.
        bg_loss_fn: Loss function for background features.
        h5_path: Path to H5 file containing features or labels.
        img_name: List of image names for saving outputs.
        save_dir: Directory to save CAMs and binary masks.
        epoch: Current epoch number for naming saved files.
    
    Returns:
        cam: Combined CAM, shape [batch_size, num_classes, h, w].
        binary_mask: Binary segmentation mask, shape [batch_size, num_classes, h, w].
        loss: Computed loss (classification + foreground/background loss).
    """
    model.eval()
    device = next(model.parameters()).device
    num_classes = 4  # Fixed number of classes, consistent with model and cls_labels

    with torch.no_grad():
        # Move inputs to device
        x = x.to(device).float()
        coords = coords.to(device).float() if coords is not None else None
        fg_bg_labels = fg_bg_labels.to(device).float() if fg_bg_labels is not None else torch.ones((x.shape[0], 1), dtype=torch.float32, device=device)
        
        # Handle cls_labels
        if cls_labels is not None:
            if cls_labels.dim() == 1:
                cls_labels = F.one_hot(cls_labels.long(), num_classes=num_classes).float().to(device)
                print(f"Converted cls_labels to one-hot: {cls_labels.shape}, dtype: {cls_labels.dtype}")
            else:
                cls_labels = cls_labels.to(device).float()
        else:
            # Load from H5 file if provided
            if h5_path and os.path.exists(h5_path):
                try:
                    with h5py.File(h5_path, 'r') as h5:
                        h5_label = h5.attrs.get('label', None)
                        if h5_label is not None:
                            cls_labels = torch.tensor(h5_label[:num_classes], dtype=torch.float32).unsqueeze(0).to(device)
                            print(f"Loaded cls_label from H5 {h5_path}: {cls_labels.shape}, value: {cls_labels}")
                        else:
                            cls_labels = torch.full((x.shape[0], num_classes), 0.25, dtype=torch.float32, device=device)
                            print(f"No label attribute found in H5 {h5_path}, defaulted cls_labels: {cls_labels.shape}")
                except Exception as e:
                    print(f"Error loading label from H5 {h5_path}: {e}")
                    cls_labels = torch.full((x.shape[0], num_classes), 0.25, dtype=torch.float32, device=device)
            else:
                cls_labels = torch.full((x.shape[0], num_classes), 0.25, dtype=torch.float32, device=device)
                print(f"Defaulted cls_labels to uniform distribution: {cls_labels.shape}")

        print(f"generate_cam - cls_labels: {cls_labels.shape}, dtype: {cls_labels.dtype}, fg_bg_labels: {fg_bg_labels.shape}, device: {fg_bg_labels.device}")

        # Generate CAMs using model.generate_cam
        cam1, cam2, cam3, cam4, loss = model.generate_cam(x, cls_labels, h5_path=h5_path, coords=coords, fg_bg_labels=fg_bg_labels)
        print(f"Generated CAM shapes: cam1={cam1.shape}, cam2={cam2.shape}, cam3={cam3.shape}, cam4={cam4.shape}")

        # Combine CAMs
        cam = torch.stack([cam1, cam2, cam3, cam4], dim=1)  # [batch_size, 4, num_classes, h, w]
        cam = torch.max(cam, dim=1)[0]  # [batch_size, num_classes, h, w]
        print(f"Combined CAM shape: {cam.shape}")

        # Resize CAM to match input dimensions
        orig_h, orig_w = x.shape[2], x.shape[3]
        if cam.shape[2:] != (orig_h, orig_w):
            cam = F.interpolate(cam, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
            print(f"cam shape after interpolate: {cam.shape}")

        # Normalize CAM and generate binary mask
        threshold = 0.1
        cam_norm = torch.zeros_like(cam)
        for c in range(num_classes):
            min_v = cam[:, c, :, :].min()
            max_v = cam[:, c, :, :].max()
            if max_v > min_v:
                cam_norm[:, c, :, :] = (cam[:, c, :, :] - min_v) / (max_v - min_v)
        cam_softmax = F.softmax(cam_norm, dim=1)
        max_values = torch.max(cam_softmax, dim=1, keepdim=True)[0]
        max_indices = torch.argmax(cam_softmax, dim=1, keepdim=True)  # [batch_size, 1, h, w]
        binary_mask = F.one_hot(max_indices.squeeze(1), num_classes=num_classes).permute(0, 3, 1, 2).float()  # [batch_size, num_classes, h, w]
        print(f"Binary mask shape: {binary_mask.shape}")

        # Compute loss using feature extractor if provided
        if feature_extractor is not None and cls_labels is not None and fg_bg_labels is not None:
            batch_info = feature_extractor.process_batch(img_name, cam, cls_labels, coords=coords, x=x)
            if batch_info is not None:
                fg_features, bg_features = batch_info['fg_features'], batch_info['bg_features']
                set_info = pair_features(fg_features, bg_features, model.l_fea, cls_labels, device=device)
                fg_features, bg_features, fg_pro, bg_pro = set_info['fg_features'], set_info['bg_features'], set_info['fg_text'], set_info['bg_text']
                print(f"fg_features shape: {fg_features.shape}, fg_pro shape: {fg_pro.shape}, bg_features shape: {bg_features.shape}, bg_pro shape: {bg_pro.shape}")
                fg_loss = fg_loss_fn(fg_features, fg_pro, bg_pro) if fg_loss_fn and fg_pro is not None else 0.0
                bg_loss = bg_loss_fn(bg_features, fg_pro, bg_pro) if bg_loss_fn and bg_pro is not None else 0.0
                # Recompute classification loss
                cls1, cls2, cls3, cls4 = cam1.mean(dim=(2, 3)), cam2.mean(dim=(2, 3)), cam3.mean(dim=(2, 3)), cam4.mean(dim=(2, 3))
                Y_prob = (cls1 + cls2 + cls3 + cls4) / 4
                cls_loss = loss_function(Y_prob, cls_labels)
                loss = cls_loss + (fg_loss + bg_loss + 0.0005 * torch.mean(cam)) * 0.1
                print(f"Computed loss: cls_loss={cls_loss.item():.4f}, fg_loss={fg_loss.item():.4f}, bg_loss={bg_loss.item():.4f}, total_loss={loss.item():.4f}")
            else:
                print("No foreground samples in batch, skipping feature-based loss")
                cls1, cls2, cls3, cls4 = cam1.mean(dim=(2, 3)), cam2.mean(dim=(2, 3)), cam3.mean(dim=(2, 3)), cam4.mean(dim=(2, 3))
                Y_prob = (cls1 + cls2 + cls3 + cls4) / 4
                loss = loss_function(Y_prob, cls_labels)
                print(f"Computed loss without features: {loss.item():.4f}")
        else:
            print("No feature extractor provided, using model loss")
            cls1, cls2, cls3, cls4 = cam1.mean(dim=(2, 3)), cam2.mean(dim=(2, 3)), cam3.mean(dim=(2, 3)), cam4.mean(dim=(2, 3))
            Y_prob = (cls1 + cls2 + cls3 + cls4) / 4
            loss = loss_function(Y_prob, cls_labels)
            print(f"Computed loss without feature extractor: {loss.item():.4f}")

        # Save CAM and binary mask if save_dir is provided
        if save_dir and img_name and epoch:
            cam_heatmap_dir = os.path.join(save_dir, "CAM_heatmap")
            binary_mask_dir = os.path.join(save_dir, "binary_mask")
            os.makedirs(cam_heatmap_dir, exist_ok=True)
            os.makedirs(binary_mask_dir, exist_ok=True)
            base_name = os.path.basename(os.path.splitext(img_name[0])[0])
            cam_heatmap_path = os.path.join(cam_heatmap_dir, f"cam_heatmap_{base_name}_epoch_{epoch}.pth")
            binary_mask_path = os.path.join(binary_mask_dir, f"binary_mask_{base_name}_epoch_{epoch}.pth")
            torch.save(cam.cpu(), cam_heatmap_path)
            torch.save(binary_mask.cpu(), binary_mask_path)
            print(f"Saved CAM heatmap: {cam_heatmap_path}, Binary Mask: {binary_mask_path}")

    return cam, binary_mask, loss
    
# def create_combined_heatmap(cam, num_classes=4):  # Changed to 4 classes
#     combined = np.sum(cam[:num_classes], axis=0)
#     combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-8)
#     return combined

def create_colored_binary_mask(cam, class_map, threshold=0.1):
    h, w = cam.shape[1], cam.shape[2]
    cam_softmax = F.softmax(torch.tensor(cam), dim=0).numpy()
    max_indices = np.argmax(cam_softmax, axis=0)
    max_values = np.max(cam_softmax, axis=0)
    mask_np = np.zeros((h, w), dtype=np.uint8)
    for c in range(4):  # Changed to 4 classes
        binary = (max_indices == c) & (max_values > threshold)
        mask_np[binary] = c
    structure = np.ones((3, 3), dtype=np.uint8)
    dilated = np.zeros_like(mask_np)
    for c in range(4):  # Changed to 4 classes
        class_bin = (mask_np == c)
        dilated_class = binary_dilation(class_bin, structure=structure, iterations=1)
        new_pixels = dilated_class & (dilated == 0)
        dilated[new_pixels] = c
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(4):  # Changed to 4 classes
        bin_m = (dilated == c)
        colored[bin_m] = np.array(class_map[c][2], dtype=np.uint8)
    return colored

def create_combined_colored_binary_mask(cam, class_map, threshold=0.1):
    h, w = cam.shape[1], cam.shape[2]
    cam_softmax = F.softmax(torch.tensor(cam), dim=0).numpy()
    max_indices = np.argmax(cam_softmax, axis=0)
    max_values = np.max(cam_softmax, axis=0)
    mask_np = np.zeros((h, w), dtype=np.uint8)
    for c in range(4):  # Changed to 4 classes
        binary = (max_indices == c) & (max_values > threshold)
        mask_np[binary] = c
    structure = np.ones((3, 3), dtype=np.uint8)
    dilated = np.zeros_like(mask_np)
    for c in range(4):  # Changed to 4 classes
        class_bin = (mask_np == c)
        dilated_class = binary_dilation(class_bin, structure=structure, iterations=1)
        new_pixels = dilated_class & (dilated == 0)
        dilated[new_pixels] = c
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(4):  # Changed to 4 classes
        bin_m = (dilated == c)
        colored[bin_m] = np.array(class_map[c][2], dtype=np.uint8)
    return colored

def create_combined_heatmap(cam_np: np.ndarray) -> np.ndarray:
    """Placeholder for combining heatmap channels. Replace with your implementation."""
    if cam_np.ndim == 3 and cam_np.shape[0] == 4:  # [4, height, width]
        return np.max(cam_np, axis=0)  # Simple max pooling across classes
    elif cam_np.ndim == 2:  # [height, width]
        return cam_np
    else:
        raise ValueError(f"Unexpected cam_np shape: {cam_np.shape}")

def visualize_original_and_combined(model, x, batch_size, save_dir, feature_extractor, loss_function, fg_loss_fn, bg_loss_fn, h5_path=None, epoch=None, gt_mask=None, img_name=None):
    """
    Visualize original image, combined CAM, and ground truth mask.
    """
    device = next(model.parameters()).device
    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        x = x.to(device).float()
        orig_h, orig_w = x.shape[2], x.shape[3]

        # Validate and process gt_mask or use cls_labels
        if gt_mask is None and hasattr(x, 'cls_labels') and x.cls_labels is not None:
            print("Warning: gt_mask is None, using cls_labels for ground truth.")
            gt_mask = x.cls_labels.to(device)
            if gt_mask.dim() < 3:
                if gt_mask.dim() == 1 or (gt_mask.dim() == 2 and gt_mask.shape[-1] <= 4):
                    dominant_class = torch.argmax(gt_mask, dim=1) if gt_mask.dim() == 2 else gt_mask
                    gt_mask = dominant_class.view(batch_size, 1, 1).expand(-1, orig_h, orig_w).to(device, dtype=torch.uint8)
                else:
                    gt_mask = gt_mask.unsqueeze(0) if gt_mask.dim() == 2 else gt_mask
            if gt_mask.shape[1:] != (orig_h, orig_w):
                print(f"Warning: cls_labels shape {gt_mask.shape[1:]} does not match image shape {orig_h, orig_w}. Interpolating.")
                gt_mask = F.interpolate(gt_mask.float().unsqueeze(1), size=(orig_h, orig_w), mode='nearest').squeeze(1).to(torch.uint8)
            gt_mask = torch.clamp(gt_mask, 0, 3)
        elif gt_mask is None:
            print("Warning: gt_mask and cls_labels are None, using zero mask with default shape.")
            gt_mask = torch.zeros((batch_size, orig_h, orig_w), device=device, dtype=torch.uint8)
        else:
            gt_mask = gt_mask.to(device)
            if gt_mask.dim() < 3:
                if gt_mask.dim() == 1 or (gt_mask.dim() == 2 and gt_mask.shape[-1] <= 4):
                    print("Warning: gt_mask is a class vector, converting to spatial mask.")
                    dominant_class = torch.argmax(gt_mask, dim=1) if gt_mask.dim() == 2 else gt_mask
                    gt_mask = dominant_class.view(batch_size, 1, 1).expand(-1, orig_h, orig_w).to(device, dtype=torch.uint8)
                else:
                    gt_mask = gt_mask.unsqueeze(0) if gt_mask.dim() == 2 else gt_mask
            if gt_mask.shape[1:] != (orig_h, orig_w):
                print(f"Warning: gt_mask shape {gt_mask.shape[1:]} does not match image shape {orig_h, orig_w}. Interpolating.")
                gt_mask = F.interpolate(gt_mask.float().unsqueeze(1), size=(orig_h, orig_w), mode='nearest').squeeze(1).to(torch.uint8)
            gt_mask = torch.clamp(gt_mask, 0, 3)
            print(f"gt_mask shape after processing: {gt_mask.shape}, min: {gt_mask.min()}, max: {gt_mask.max()}")

        # Generate CAM
        cam, binary_mask, _ = generate_cam(
            model, x, None, None, None, feature_extractor=feature_extractor, 
            loss_function=loss_function, fg_loss_fn=fg_loss_fn, bg_loss_fn=bg_loss_fn, 
            h5_path=h5_path, img_name=img_name, save_dir=save_dir, epoch=epoch
        )
        if cam is None or binary_mask is None:
            print("Error: CAM or binary mask is None, using zero mask as fallback.")
            gt_mask = torch.zeros((batch_size, orig_h, orig_w), device=device, dtype=torch.uint8)
        elif gt_mask is None:
            print("Warning: gt_mask is None, using binary_mask as fallback.")
            gt_mask = binary_mask.to(device, dtype=torch.uint8)

        # Process each sample in the batch
        for sample_idx in range(min(batch_size, 1)):
            inputs_cpu = x[sample_idx].cpu()
            cam_cpu = cam[sample_idx].cpu()
            gt_mask_cpu = gt_mask[sample_idx].cpu()
            binary_mask_cpu = binary_mask[sample_idx].cpu()

            inputs_np = inputs_cpu.numpy().transpose(1, 2, 0)
            cam_np = cam_cpu.numpy().transpose(1, 2, 0) if cam_cpu.ndim == 3 else cam_cpu.numpy()[None].transpose(1, 2, 0)
            cam_np_reduced = np.max(cam_np, axis=2, keepdims=True)
            cam_np_reduced = np.repeat(cam_np_reduced, 3, axis=2)

            gt_mask_np = gt_mask_cpu.numpy()
            binary_mask_np = binary_mask_cpu.numpy().transpose(1, 2, 0) if binary_mask_cpu.ndim == 4 else binary_mask_cpu.numpy()[..., None]

            colored_gt = np.zeros((gt_mask_np.shape[0], gt_mask_np.shape[1], 3), dtype=np.uint8)
            for c in range(4):
                valid_mask = (gt_mask_np == c)
                if np.any(valid_mask):
                    colored_gt[valid_mask] = np.array(class_descriptions[c][2], dtype=np.uint8)

            combined = inputs_np * 0.5 + cam_np_reduced * 0.5
            combined = np.clip(combined, 0, 1)

            fig = plt.figure(figsize=(15, 5))
            gs = fig.add_gridspec(1, 3, wspace=0.3)
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.imshow(inputs_np)
            ax1.set_title('Original Image')
            ax1.axis('off')
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.imshow(combined)
            ax2.set_title('Combined CAM')
            ax2.axis('off')
            ax3 = fig.add_subplot(gs[0, 2])
            ax3.imshow(colored_gt)
            ax3.set_title('Ground Truth Mask')
            ax3.axis('off')

            save_path = os.path.join(save_dir, f'original_combined_epoch_{epoch}_sample_{sample_idx}.png') if epoch else os.path.join(save_dir, f'original_combined_sample_{sample_idx}.png')
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close()
            print(f"Saved original and combined visualization: {save_path}")
            
def visualize_original_and_binary_masks(model, x, batch_size, save_dir, feature_extractor, loss_function, fg_loss_fn, bg_loss_fn, h5_path=None, epoch=None, gt_mask=None, img_name=None, class_map=class_descriptions):
    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        x = x.to(next(model.parameters()).device).float()
        cam, binary_mask, _ = generate_cam(
            model, x, None, None, None, feature_extractor=feature_extractor, 
            loss_function=loss_function, fg_loss_fn=fg_loss_fn, bg_loss_fn=bg_loss_fn, 
            h5_path=h5_path, img_name=img_name, epoch=epoch
        )
        if cam is None or binary_mask is None:
            print("Error: CAM or binary mask is None, skipping visualization.")
            return

    for sample_idx in range(min(batch_size, 1)):
        low_img = x[sample_idx].cpu().permute(1, 2, 0).numpy()
        low_img = (low_img - low_img.min()) / (low_img.max() - low_img.min() + 1e-8)
        cam_np = cam[sample_idx].cpu().numpy()
        binary_mask_np = binary_mask[sample_idx].cpu().numpy()

        if binary_mask_np.shape[0] == 4:  # [4, height, width]
            binary_mask_np = np.argmax(binary_mask_np, axis=0)
        elif binary_mask_np.ndim == 3 and binary_mask_np.shape[0] == 1:
            binary_mask_np = binary_mask_np.squeeze(0)
        elif binary_mask_np.ndim != 2:
            print(f"Error: Unexpected binary_mask shape {binary_mask_np.shape}, skipping sample.")
            continue

        h, w = binary_mask_np.shape
        structure = np.ones((3, 3), dtype=np.uint8)
        dilated_mask = np.zeros((h, w), dtype=np.uint8)
        for c in range(4):
            class_bin = (binary_mask_np == c)
            dilated_class = binary_dilation(class_bin, structure=structure, iterations=1)
            new_pixels = dilated_class & (dilated_mask == 0)
            dilated_mask[new_pixels] = c

        colored_binary = np.zeros((h, w, 3), dtype=np.uint8)
        for c in range(4):
            mask = (dilated_mask == c)
            if np.any(mask):
                colored_binary[mask] = np.array(class_map[c][2], dtype=np.uint8)

        # Process gt_mask or use cls_labels
        if gt_mask is not None:
            gt_mask_cpu = gt_mask[sample_idx].cpu()
            if gt_mask_cpu.dim() == 2 and gt_mask_cpu.shape == (h, w):
                gt_mask_spatial = gt_mask_cpu
            elif gt_mask_cpu.dim() == 1 or (gt_mask_cpu.dim() == 2 and gt_mask_cpu.shape[-1] <= 4):
                dominant_class = torch.argmax(gt_mask_cpu).item() if gt_mask_cpu.dim() == 1 else torch.argmax(gt_mask_cpu).item()
                gt_mask_spatial = torch.full((h, w), dominant_class, dtype=torch.uint8)
            else:
                print(f"Warning: Unexpected gt_mask shape {gt_mask_cpu.shape}, using cls_labels.")
                gt_mask_spatial = None
        else:
            print("Warning: gt_mask is None, using cls_labels for ground truth.")
            gt_mask_spatial = None

        if gt_mask_spatial is None and hasattr(x, 'cls_labels') and x.cls_labels is not None:
            cls_labels_cpu = x.cls_labels[sample_idx].cpu()
            if cls_labels_cpu.dim() == 1 or (cls_labels_cpu.dim() == 2 and cls_labels_cpu.shape[-1] <= 4):
                dominant_class = torch.argmax(cls_labels_cpu).item() if cls_labels_cpu.dim() == 1 else torch.argmax(cls_labels_cpu).item()
                gt_mask_spatial = torch.full((h, w), dominant_class, dtype=torch.uint8)
            else:
                print(f"Warning: Unexpected cls_labels shape {cls_labels_cpu.shape}, using zero mask.")
                gt_mask_spatial = torch.zeros((h, w), dtype=torch.uint8)

        colored_gt = np.zeros((h, w, 3), dtype=np.uint8)
        if gt_mask_spatial is not None:
            gt_mask_np = gt_mask_spatial.numpy()
            for c in range(4):
                colored_gt[gt_mask_np == c] = np.array(class_map[c][2], dtype=np.uint8)

        fig = plt.figure(figsize=(15, 5))
        gs = fig.add_gridspec(1, 3, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(low_img)
        ax1.set_title('Original Resolution')
        ax1.axis('off')
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(colored_binary)
        ax2.set_title('Colored Binary Mask')
        ax2.axis('off')
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(colored_gt)
        ax3.set_title('Ground Truth Mask')
        ax3.axis('off')

        save_path = os.path.join(save_dir, f'original_binary_masks_epoch_{epoch}_sample_{sample_idx}.png') if epoch else os.path.join(save_dir, f'original_binary_masks_sample_{sample_idx}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Saved original and colored binary masks: {save_path}")

def visualize_original_and_combined_heatmap(model, x, batch_size, save_dir, feature_extractor, loss_function, fg_loss_fn, bg_loss_fn, h5_path=None, epoch=None, gt_mask=None, img_name=None, class_map=class_descriptions):
    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        x = x.to(next(model.parameters()).device).float()
        cam, _, _ = generate_cam(
            model, x, None, None, None, feature_extractor=feature_extractor, 
            loss_function=loss_function, fg_loss_fn=fg_loss_fn, bg_loss_fn=bg_loss_fn, 
            h5_path=h5_path, img_name=img_name, epoch=epoch
        )
        if cam is None:
            print("Error: CAM is None, skipping visualization.")
            return
    
    for sample_idx in range(min(batch_size, 1)):
        low_img = x[sample_idx].cpu().permute(1, 2, 0).numpy()
        low_img = (low_img - low_img.min()) / (low_img.max() - low_img.min() + 1e-8)
        cam_np = cam[sample_idx].cpu().numpy()
        combined_heatmap = create_combined_heatmap(cam_np)

        h, w = combined_heatmap.shape[:2]
        # Process gt_mask or use cls_labels
        if gt_mask is not None:
            gt_mask_cpu = gt_mask[sample_idx].cpu()
            if gt_mask_cpu.dim() == 2 and gt_mask_cpu.shape == (h, w):
                gt_mask_spatial = gt_mask_cpu
            elif gt_mask_cpu.dim() == 1 or (gt_mask_cpu.dim() == 2 and gt_mask_cpu.shape[-1] <= 4):
                dominant_class = torch.argmax(gt_mask_cpu).item() if gt_mask_cpu.dim() == 1 else torch.argmax(gt_mask_cpu).item()
                gt_mask_spatial = torch.full((h, w), dominant_class, dtype=torch.uint8)
            else:
                print(f"Warning: Unexpected gt_mask shape {gt_mask_cpu.shape}, using cls_labels.")
                gt_mask_spatial = None
        else:
            print("Warning: gt_mask is None, using cls_labels for ground truth.")
            gt_mask_spatial = None

        if gt_mask_spatial is None and hasattr(x, 'cls_labels') and x.cls_labels is not None:
            cls_labels_cpu = x.cls_labels[sample_idx].cpu()
            if cls_labels_cpu.dim() == 1 or (cls_labels_cpu.dim() == 2 and cls_labels_cpu.shape[-1] <= 4):
                dominant_class = torch.argmax(cls_labels_cpu).item() if cls_labels_cpu.dim() == 1 else torch.argmax(cls_labels_cpu).item()
                gt_mask_spatial = torch.full((h, w), dominant_class, dtype=torch.uint8)
            else:
                print(f"Warning: Unexpected cls_labels shape {cls_labels_cpu.shape}, using zero mask.")
                gt_mask_spatial = torch.zeros((h, w), dtype=torch.uint8)

        colored_gt = np.zeros((h, w, 3), dtype=np.uint8)
        if gt_mask_spatial is not None:
            gt_mask_np = gt_mask_spatial.numpy()
            for c in range(4):
                colored_gt[gt_mask_np == c] = np.array(class_map[c][2], dtype=np.uint8)

        fig = plt.figure(figsize=(15, 5))
        gs = fig.add_gridspec(1, 3, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(low_img)
        ax1.set_title('Original Resolution')
        ax1.axis('off')
        ax2 = fig.add_subplot(gs[0, 1])
        im = ax2.imshow(combined_heatmap, cmap='jet')
        ax2.set_title('Combined Heatmap')
        plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        ax2.axis('off')
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(colored_gt)
        ax3.set_title('Ground Truth Mask')
        ax3.axis('off')

        save_path = os.path.join(save_dir, f'original_combined_heatmap_epoch_{epoch}_sample_{sample_idx}.png') if epoch else os.path.join(save_dir, f'original_combined_heatmap_sample_{sample_idx}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Saved original and combined heatmap: {save_path}")

def visualize_original_and_heatmaps(model, x, batch_size, save_dir, feature_extractor, loss_function, fg_loss_fn, bg_loss_fn, h5_path=None, epoch=None, gt_mask=None, img_name=None, class_map=class_descriptions):
    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        x = x.to(next(model.parameters()).device).float()
        cam, _, _ = generate_cam(
            model, x, None, None, None, feature_extractor=feature_extractor, 
            loss_function=loss_function, fg_loss_fn=fg_loss_fn, bg_loss_fn=bg_loss_fn, 
            h5_path=h5_path, img_name=img_name, epoch=epoch
        )
        if cam is None:
            print("Error: CAM is None, skipping visualization.")
            return
    
    for sample_idx in range(min(batch_size, 1)):
        low_img = x[sample_idx].cpu().permute(1, 2, 0).numpy()
        low_img = (low_img - low_img.min()) / (low_img.max() - low_img.min() + 1e-8)
        cam_np = cam[sample_idx].cpu().numpy()

        # Correct dimension unpacking
        if cam_np.ndim == 3 and cam_np.shape[0] == 4:  # [4, height, width]
            h, w = cam_np.shape[1], cam_np.shape[2]
        else:  # [height, width]
            h, w = cam_np.shape[0], cam_np.shape[1]

        fig = plt.figure(figsize=(25, 5))
        gs = fig.add_gridspec(1, 6, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(low_img)
        ax1.set_title('Original Resolution')
        ax1.axis('off')
        for c in range(4):
            ax = fig.add_subplot(gs[0, c + 1])
            cam_channel = cam_np[c] if cam_np.ndim == 3 and cam_np.shape[0] == 4 else cam_np
            cam_norm = (cam_channel - cam_channel.min()) / (cam_channel.max() - cam_channel.min() + 1e-8)
            im = ax.imshow(cam_norm, cmap='jet')
            ax.set_title(f'Heatmap {class_map[c][1]}')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.axis('off')
        # Use cls_labels as ground truth if gt_mask is None
        if gt_mask is not None:
            gt_mask_cpu = gt_mask[sample_idx].cpu()
            if gt_mask_cpu.dim() == 2 and gt_mask_cpu.shape == (h, w):
                gt_mask_spatial = gt_mask_cpu
            elif gt_mask_cpu.dim() == 1 or (gt_mask_cpu.dim() == 2 and gt_mask_cpu.shape[-1] <= 4):
                dominant_class = torch.argmax(gt_mask_cpu).item() if gt_mask_cpu.dim() == 1 else torch.argmax(gt_mask_cpu).item()
                gt_mask_spatial = torch.full((h, w), dominant_class, dtype=torch.uint8)
            else:
                print(f"Warning: Unexpected gt_mask shape {gt_mask_cpu.shape}, using cls_labels.")
                gt_mask_spatial = None
        else:
            print("Warning: gt_mask is None, using cls_labels for ground truth.")
            gt_mask_spatial = None

        if gt_mask_spatial is None and hasattr(x, 'cls_labels') and x.cls_labels is not None:
            cls_labels_cpu = x.cls_labels[sample_idx].cpu()
            if cls_labels_cpu.dim() == 1 or (cls_labels_cpu.dim() == 2 and cls_labels_cpu.shape[-1] <= 4):
                dominant_class = torch.argmax(cls_labels_cpu).item() if cls_labels_cpu.dim() == 1 else torch.argmax(cls_labels_cpu).item()
                gt_mask_spatial = torch.full((h, w), dominant_class, dtype=torch.uint8)
            else:
                print(f"Warning: Unexpected cls_labels shape {cls_labels_cpu.shape}, using zero mask.")
                gt_mask_spatial = torch.zeros((h, w), dtype=torch.uint8)

        colored_gt = np.zeros((h, w, 3), dtype=np.uint8)
        if gt_mask_spatial is not None:
            gt_mask_np = gt_mask_spatial.numpy()
            for c in range(4):
                colored_gt[gt_mask_np == c] = np.array(class_map[c][2], dtype=np.uint8)
        ax5 = fig.add_subplot(gs[0, 5])
        ax5.imshow(colored_gt)
        ax5.set_title('Ground Truth Mask')
        ax5.axis('off')

        save_path = os.path.join(save_dir, f'original_heatmaps_epoch_{epoch}_sample_{sample_idx}.png') if epoch else os.path.join(save_dir, f'original_heatmaps_sample_{sample_idx}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Saved original and heatmaps: {save_path}")

def visualize_original_and_binary_per_class(model, x, batch_size, save_dir, feature_extractor, loss_function, fg_loss_fn, bg_loss_fn, h5_path=None, epoch=None, gt_mask=None, img_name=None, class_map=class_descriptions):
    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        x = x.to(next(model.parameters()).device).float()
        cam, binary_mask, _ = generate_cam(
            model, x, None, None, None, feature_extractor=feature_extractor, 
            loss_function=loss_function, fg_loss_fn=fg_loss_fn, bg_loss_fn=bg_loss_fn, 
            h5_path=h5_path, img_name=img_name, epoch=epoch
        )
        if cam is None or binary_mask is None:
            print("Error: CAM or binary mask is None, skipping visualization.")
            return
    
    for sample_idx in range(min(batch_size, 1)):
        low_img = x[sample_idx].cpu().permute(1, 2, 0).numpy()
        low_img = (low_img - low_img.min()) / (low_img.max() - low_img.min() + 1e-8)
        binary_mask_np = binary_mask[sample_idx].cpu().numpy()

        if binary_mask_np.shape[0] == 4:  # [4, height, width]
            binary_mask_np = np.argmax(binary_mask_np, axis=0)
        elif binary_mask_np.ndim == 3 and binary_mask_np.shape[0] == 1:
            binary_mask_np = binary_mask_np.squeeze(0)
        elif binary_mask_np.ndim != 2:
            print(f"Error: Unexpected binary_mask shape {binary_mask_np.shape}, skipping sample.")
            continue

        h, w = binary_mask_np.shape
        fig = plt.figure(figsize=(25, 5))
        gs = fig.add_gridspec(1, 6, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(low_img)
        ax1.set_title('Original Resolution')
        ax1.axis('off')
        for c in range(4):
            ax = fig.add_subplot(gs[0, c + 1])
            class_bin = np.zeros((h, w, 3), dtype=np.uint8)
            mask = (binary_mask_np == c)
            if np.any(mask):
                class_bin[mask] = np.array(class_map[c][2], dtype=np.uint8)
            ax.imshow(class_bin)
            ax.set_title(f'Binary Mask {class_map[c][1]}')
            ax.axis('off')
        # Process gt_mask or use cls_labels
        if gt_mask is not None:
            gt_mask_cpu = gt_mask[sample_idx].cpu()
            if gt_mask_cpu.dim() == 2 and gt_mask_cpu.shape == (h, w):
                gt_mask_spatial = gt_mask_cpu
            elif gt_mask_cpu.dim() == 1 or (gt_mask_cpu.dim() == 2 and gt_mask_cpu.shape[-1] <= 4):
                dominant_class = torch.argmax(gt_mask_cpu).item() if gt_mask_cpu.dim() == 1 else torch.argmax(gt_mask_cpu).item()
                gt_mask_spatial = torch.full((h, w), dominant_class, dtype=torch.uint8)
            else:
                print(f"Warning: Unexpected gt_mask shape {gt_mask_cpu.shape}, using cls_labels.")
                gt_mask_spatial = None
        else:
            print("Warning: gt_mask is None, using cls_labels for ground truth.")
            gt_mask_spatial = None

        if gt_mask_spatial is None and hasattr(x, 'cls_labels') and x.cls_labels is not None:
            cls_labels_cpu = x.cls_labels[sample_idx].cpu()
            if cls_labels_cpu.dim() == 1 or (cls_labels_cpu.dim() == 2 and cls_labels_cpu.shape[-1] <= 4):
                dominant_class = torch.argmax(cls_labels_cpu).item() if cls_labels_cpu.dim() == 1 else torch.argmax(cls_labels_cpu).item()
                gt_mask_spatial = torch.full((h, w), dominant_class, dtype=torch.uint8)
            else:
                print(f"Warning: Unexpected cls_labels shape {cls_labels_cpu.shape}, using zero mask.")
                gt_mask_spatial = torch.zeros((h, w), dtype=torch.uint8)

        colored_gt = np.zeros((h, w, 3), dtype=np.uint8)
        if gt_mask_spatial is not None:
            gt_mask_np = gt_mask_spatial.numpy()
            for c in range(4):
                colored_gt[gt_mask_np == c] = np.array(class_map[c][2], dtype=np.uint8)
        ax5 = fig.add_subplot(gs[0, 5])
        ax5.imshow(colored_gt)
        ax5.set_title('Ground Truth Mask')
        ax5.axis('off')

        save_path = os.path.join(save_dir, f'original_binary_per_class_epoch_{epoch}_sample_{sample_idx}.png') if epoch else os.path.join(save_dir, f'original_binary_per_class_sample_{sample_idx}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Saved original and binary masks per class: {save_path}")
              
# Validation
def batch_iou(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute the mean IoU for a batch of segmentation predictions.
    
    Args:
        pred (torch.Tensor): Predicted segmentation masks, shape [batch_size, height, width]
        target (torch.Tensor): Ground truth masks, shape [batch_size, height, width]
    
    Returns:
        torch.Tensor: Mean IoU across the batch
    """
    assert pred.shape == target.shape, f"Shape mismatch: pred {pred.shape} vs target {target.shape}"
    batch_size = pred.shape[0]
    iou_scores = []

    for i in range(batch_size):
        pred_i = pred[i].cpu().numpy()
        target_i = target[i].cpu().numpy()
        
        # Ensure unique class indices are handled (0 to max class)
        unique_classes = np.unique(np.concatenate((pred_i, target_i)))
        iou_per_class = []
        
        for c in unique_classes:
            if c == 0:  # Skip background if not of interest
                continue
            pred_mask = (pred_i == c)
            target_mask = (target_i == c)
            intersection = np.logical_and(pred_mask, target_mask).sum()
            union = np.logical_or(pred_mask, target_mask).sum()
            if union == 0:
                iou = 1.0 if intersection == 0 else 0.0  # Handle empty masks
            else:
                iou = intersection / union
            iou_per_class.append(iou)
        
        if iou_per_class:
            iou_scores.append(np.mean(iou_per_class))
        else:
            iou_scores.append(0.0)  # Default to 0 if no valid classes

    return torch.tensor(iou_scores).mean().to(pred.device)

def validate(model, val_loader, cfg, loss_function, miou_metric, fw_iou_metric, b_iou_metric, dice_metric, epoch, h5_val_dir, feature_extractor, gt_mask=None):
    device = next(model.parameters()).device
    model.eval()
    total_loss = 0.0
    iters_per_epoch = len(val_loader)
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for n_iter, batch in enumerate(tqdm(val_loader, desc=f"Validation Epoch {epoch}", total=iters_per_epoch)):
            h5_paths, inputs, cls_labels, masks = batch
            inputs = inputs.to(device).float()
            cls_labels = cls_labels.to(device)
            coords = None
            for h5_path in h5_paths:
                corrected_h5_path = os.path.join(os.path.dirname(h5_path), os.path.basename(h5_path))
                if os.path.exists(corrected_h5_path):
                    try:
                        with h5py.File(corrected_h5_path, 'r') as h5:
                            coords_data = h5['coords'][:] if 'coords' in h5 else None
                            if coords_data is not None:
                                coords = torch.from_numpy(coords_data).float().to(device)
                                if coords.dim() == 1 and coords.shape[0] == 2:
                                    coords = coords.unsqueeze(0).expand(inputs.size(0), -1)
                                elif coords.shape[0] != inputs.size(0) and coords.shape[0] == 1:
                                    coords = coords.expand(inputs.size(0), -1)
                    except Exception as e:
                        print(f"Error loading H5 file {corrected_h5_path}: {e}")
                        continue
                    break
            fg_bg_labels = torch.ones((inputs.size(0), 1), dtype=torch.float32).to(device)

            # Forward pass through the model
            cls1, cam1, cls2, cam2, cls3, cam3, cls4, cam4, l_fea, _ = model(
                inputs, labels=cls_labels, h5_path=corrected_h5_path if os.path.exists(corrected_h5_path) else None,
                coords=coords, fg_bg_labels=fg_bg_labels
            )

            # Debug shapes
            print(f"cam1 shape: {cam1.shape}, cam2 shape: {cam2.shape}, cam3 shape: {cam3.shape}, cam4 shape: {cam4.shape}")
            print(f"Y_prob shape (ClsNetwork): {(cls1 + cls2 + cls3 + cls4) / 4}.shape")

            # Adjust cam construction
            if cam1.dim() == 3 and cam1.shape[1] == 4:
                cam = torch.stack([cam1, cam2, cam3, cam4], dim=1)
            else:
                cam = torch.stack([cam1, cam2, cam3, cam4], dim=0).mean(dim=0)
                if cam.dim() == 3 and cam.shape[1] != 4:
                    num_classes = 4
                    cam = cam.unsqueeze(1).repeat(1, num_classes, 1, 1)
            print(f"Constructed cam shape: {cam.shape}")

            # Extract features
            features = feature_extractor(inputs, img_name=h5_paths, cam=cam, label=cls_labels, coords=coords)
            if features.dim() != 4:
                print(f"Warning: features shape {features.shape} is not 4D, reshaping to [batch_size, 128, height, width]")
                orig_h, orig_w = inputs.shape[2], inputs.shape[3]
                features = features.view(inputs.size(0), 128, 1, 1).expand(inputs.size(0), 128, orig_h, orig_w)

            # Compute Y_prob
            Y_prob = (cls1 + cls2 + cls3 + cls4) / 4
            orig_h, orig_w = inputs.shape[2], inputs.shape[3]
            Y_prob_expanded = Y_prob.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, orig_h, orig_w)

            # Average spatial dimensions to match cls_labels shape [1, 4]
            Y_prob_averaged = Y_prob_expanded.mean(dim=[2, 3])  # [1, 4]

            # Combine features for potential other use
            combined_input = torch.cat((Y_prob_expanded, features), dim=1)

            # Compute loss using cross_entropy_loss for multi-class classification
            import torch.nn.functional as F
            loss = F.cross_entropy(Y_prob_averaged, cls_labels.argmax(dim=1))  # Use class indices
            print(f"Computed loss with feature extractor: {loss.item():.4f}")

            # Rest of the validation logic...
            cam = torch.stack([cam1, cam2, cam3, cam4], dim=1) if cam1.dim() == 3 and cam1.shape[1] == 4 else torch.max(torch.stack([cam1, cam2, cam3, cam4], dim=0), dim=0)[0]
            binary_mask = (torch.sigmoid(cam) > 0.1).float()

            if masks is None:
                seg_pred = Y_prob.argmax(dim=1)
            else:
                seg_pred = binary_mask.argmax(dim=1)

            target = cls_labels.argmax(dim=1) if masks is None else masks.to(device).argmax(dim=1)
            target_shape = target.shape if target.dim() == 3 else (target.shape[0], inputs.shape[2], inputs.shape[3])

            if seg_pred.shape[1:] != target_shape[1:]:
                seg_pred = F.interpolate(seg_pred.float().unsqueeze(1), size=target_shape[1:], mode='nearest').squeeze(1).long()

            miou_metric.update(seg_pred, target)
            fw_iou_metric.update(frequency_weighted_iou(seg_pred, target).cpu().item())
            b_iou_metric.update(batch_iou(seg_pred, target).cpu().item())
            dice_metric.update(seg_pred, target)
            total_loss += loss.item()

            all_preds.append(seg_pred.cpu())
            all_labels.append(target.cpu())

    avg_loss = total_loss / iters_per_epoch
    miou_score = miou_metric.compute()
    dice_score = dice_metric.compute()
    all_preds_tensor = torch.cat(all_preds, dim=0)
    all_labels_tensor = torch.cat(all_labels, dim=0)
    all_cls_acc4 = ((all_preds_tensor == all_labels_tensor).float().mean() * 100).item()
    avg_cls_acc4 = (((all_preds_tensor == all_labels_tensor).float().mean(dim=0)).mean() * 100).item()

    miou_metric.reset()
    dice_metric.reset()
    fw_iou_metric.reset()
    b_iou_metric.reset()

    return all_cls_acc4, avg_cls_acc4, miou_score, fw_iou_metric, b_iou_metric, dice_score, avg_loss

def train(cfg, model, optimizer, feature_extractor, loss_function, fg_loss_fn, bg_loss_fn, miou_metric, dice_metric, device, train_loader, val_loader, test_loader, ckpt_dir, pred_dir, csv_dir, start_epoch=0, best_miou=0.0):
    # Ensure feature_extractor is initialized with the correct class
    if not hasattr(feature_extractor, 'forward') or 'img_name' not in feature_extractor.forward.__code__.co_varnames:
        print("Warning: feature_extractor might not have the updated forward method. Reinitializing...")
        from utils.fgbg_feature import FeatureExtractor, MaskAdapter_DynamicThreshold
        mask_adapter = MaskAdapter_DynamicThreshold(alpha=0.5)
        feature_extractor = FeatureExtractor(mask_adapter, h5_path=cfg.work_dir.h5_dir, device=device)
        
    global iters_per_epoch, start_time
    start_time = datetime.datetime.now()
    iters_per_epoch = len(train_loader)
    max_iters = cfg.train.epoch * iters_per_epoch
    warmup_iter = cfg.scheduler.warmup_iter * iters_per_epoch if hasattr(cfg.scheduler, 'warmup_iter') else 0

    csv_file = os.path.join(csv_dir, f"training_log_{start_time.strftime('%Y%m%d_%H%M%S')}.csv")
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Iteration", "Loss", "Accuracy", "Validation_mIoU",
                        "Validation_Dice", "Test_mIoU", "Test_Dice", "Best_mIoU"])

    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available(), init_scale=2.**16)
    time0 = datetime.datetime.now().replace(microsecond=0)

    miou_metric = miou_metric.to(device)
    dice_metric = dice_metric.to(device)
    fw_iou_metric = AverageMeter()
    b_iou_metric = AverageMeter()

    for epoch in range(start_epoch, cfg.train.epoch):
        model.train()
        total_loss = 0.0
        for n_iter, batch in enumerate(tqdm(train_loader, desc=f"Stage 1 Epoch {epoch+1}/{cfg.train.epoch}", total=iters_per_epoch)):
            h5_paths, inputs, cls_labels, masks = batch
            inputs = inputs.to(device).float()
            cls_labels = cls_labels.to(device)  # Ensure cls_labels is on GPU
            coords = None
            for h5_path in h5_paths:
                corrected_h5_path = os.path.join(os.path.dirname(h5_path), os.path.basename(h5_path))
                if os.path.exists(corrected_h5_path):
                    try:
                        with h5py.File(corrected_h5_path, 'r') as h5:
                            coords_data = h5['coords'][:] if 'coords' in h5 else None
                            if coords_data is not None:
                                coords = torch.from_numpy(coords_data).float().to(device)
                                if coords.dim() == 1 and coords.shape[0] == 2:
                                    coords = coords.unsqueeze(0).expand(inputs.size(0), -1)
                                elif coords.shape[0] != inputs.size(0) and coords.shape[0] == 1:
                                    coords = coords.expand(inputs.size(0), -1)
                    except Exception as e:
                        print(f"Error loading H5 file {corrected_h5_path}: {e}")
                        continue
                    break
            fg_bg_labels = torch.ones((inputs.size(0), 1), dtype=torch.float32).to(device)

            with torch.autograd.set_detect_anomaly(True):
                with torch.amp.autocast('cuda', enabled=False):
                    img_name = [os.path.basename(path) for path in h5_paths] if h5_paths else [f"image_{i}" for i in range(inputs.size(0))]
                    cls1, cam1, cls2, cam2, cls3, cam3, cls4, cam4, l_fea, loss = model(
                        inputs, labels=cls_labels, h5_path=corrected_h5_path if os.path.exists(corrected_h5_path) else None,
                        coords=coords, fg_bg_labels=fg_bg_labels
                    )
                    cls4_pred = (torch.sigmoid(cls4) > 0.1).float()
                    cam = torch.stack([cam1, cam2, cam3, cam4], dim=1)
                    cam = torch.max(cam, dim=1)[0]
                    batch_info = feature_extractor.process_batch(img_name, cam, cls_labels, coords=coords, x=inputs)
                    if batch_info is None:
                        print("No foreground samples in batch, skipping iteration.")
                        continue
                    fg_features, bg_features = batch_info['fg_features'], batch_info['bg_features']
                    set_info = pair_features(fg_features.clone(), bg_features.clone(), l_fea.clone(), cls_labels, device=device)
                    fg_features, bg_features, fg_pro, bg_pro = set_info['fg_features'], set_info['bg_features'], set_info['fg_text'], set_info['bg_text']
                    if fg_features.size(0) != fg_pro.size(0):
                        if fg_features.size(0) > fg_pro.size(0):
                            fg_pro = fg_pro.repeat(fg_features.size(0) // fg_pro.size(0) + 1, 1)[:fg_features.size(0)]
                        elif fg_features.size(0) < fg_pro.size(0):
                            fg_features = fg_features.repeat(fg_pro.size(0) // fg_features.size(0) + 1, 1)[:fg_pro.size(0)]
                    if bg_features.size(0) != bg_pro.size(0):
                        if bg_features.size(0) > bg_pro.size(0):
                            bg_pro = bg_pro.repeat(bg_features.size(0) // bg_pro.size(0) + 1, 1)[:bg_features.size(0)]
                        elif bg_features.size(0) < bg_pro.size(0):
                            bg_features = bg_features.repeat(bg_pro.size(0) // bg_features.size(0) + 1, 1)[:bg_pro.size(0)]
                    fg_features_clone = fg_features.clone().detach()
                    fg_pro_clone = fg_pro.clone().detach()
                    bg_features_clone = bg_features.clone().detach()
                    bg_pro_clone = bg_pro.clone().detach()
                    fg_loss = fg_loss_fn(fg_features_clone, fg_pro_clone, bg_pro_clone) if fg_pro_clone is not None else 0.0
                    bg_loss = bg_loss_fn(bg_features_clone, fg_pro_clone, bg_pro_clone) if bg_pro_clone is not None else 0.0
                    loss1 = loss_function(cls1, cls_labels)
                    loss2 = loss_function(cls2, cls_labels)
                    loss3 = loss_function(cls3, cls_labels)
                    loss4 = loss_function(cls4, cls_labels)
                    cls_loss = 0.25 * (loss1 + loss2 + loss3 + loss4)
                    loss = cls_loss + (fg_loss + bg_loss + 0.0005 * torch.mean(cam)) * 0.1
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"Warning: Skipping iteration due to NaN or inf in loss: {loss}")
                        continue
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item()

            if (n_iter + 1) % 100 == 0 or (n_iter + 1) == max_iters:
                delta, eta = cal_eta(time0, n_iter + 1, max_iters)
                cur_lr = optimizer.param_groups[0]['lr']
                cls_pred4 = (torch.sigmoid(cls4) > 0.1).float()
                all_cls_acc4 = (cls_pred4 == cls_labels).all(dim=1).float().mean() * 100
                avg_cls_acc4 = ((cls_pred4 == cls_labels).float().mean(dim=0)).mean() * 100
                print(f"Stage 1 - Epoch: {epoch + 1}/{cfg.train.epoch}; Iter: {n_iter + 1}/{max_iters}; "
                      f"Elapsed: {delta}; ETA: {eta}; LR: {cur_lr:.3e}; Loss: {loss.item():.4f}; "
                      f"Acc4: {all_cls_acc4:.2f}/{avg_cls_acc4:.2f}")
                with open(csv_file, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch + 1, n_iter + 1, loss.item(), all_cls_acc4.item(), "", "", "", "", best_miou])

        checkpoint_path = os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch + 1}_stage1.pth")
        torch.save({"cfg": cfg, "epoch": epoch + 1, "iter": n_iter, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "best_miou": best_miou}, checkpoint_path)
        print(f"Saved checkpoint Stage 1 at epoch {epoch + 1}: {checkpoint_path}")

        timestamp = start_time.strftime('%Y-%m-%d-%H-%M')
        cam_heatmap_dir = os.path.join(pred_dir, timestamp, "CAM_heatmap")
        binary_mask_dir = os.path.join(pred_dir, timestamp, "binary_mask")
        os.makedirs(cam_heatmap_dir, exist_ok=True)
        os.makedirs(binary_mask_dir, exist_ok=True)

        for img_name, inputs, cls_labels, masks in train_loader:
            h5_base_dir = H5_TRAIN_DIR
            h5_paths = [os.path.join(h5_base_dir, os.path.splitext(os.path.basename(name))[0] + '.h5') for name in img_name]
            inputs = inputs.to(device).float()
            cls_labels = cls_labels.to(device)
            labels_indices = cls_labels.argmax(dim=1).to(device)
            coords = None
            for h5_path in h5_paths:
                if os.path.exists(h5_path):
                    try:
                        with h5py.File(h5_path, 'r') as h5:
                            coords_data = h5['coords'][:] if 'coords' in h5 else None
                            if coords_data is not None:
                                coords = torch.from_numpy(coords_data).float().to(device)
                                if coords.dim() == 1 and coords.shape[0] == 2:
                                    coords = coords.unsqueeze(0).expand(inputs.size(0), -1)
                                elif coords.shape[0] != inputs.size(0) and coords.shape[0] == 1:
                                    coords = coords.expand(inputs.size(0), -1)
                    except Exception as e:
                        print(f"Error loading H5 file {h5_path}: {e}")
                        continue
                    break
            fg_bg_labels = torch.ones((inputs.size(0), 1), dtype=torch.float32).to(device)
            cam, binary_mask, _ = generate_cam(
                model, inputs, coords, labels_indices, fg_bg_labels=fg_bg_labels,
                feature_extractor=feature_extractor, loss_function=loss_function,
                fg_loss_fn=fg_loss_fn, bg_loss_fn=bg_loss_fn,
                h5_path=h5_paths[0] if h5_paths else None, img_name=img_name,
                save_dir=pred_dir, epoch=epoch + 1
            )

            # Generate gt_mask from cls_labels since no mask folder exists
            gt_mask = None
            if cls_labels is not None:
                dominant_class = cls_labels.argmax(dim=1)[0].item()  # Use the dominant class
                gt_mask = torch.full((1, inputs.shape[2], inputs.shape[3]), dominant_class, dtype=torch.uint8, device=device)
                print(f"Generated gt_mask from cls_labels for {img_name[0]}: shape {gt_mask.shape}, value: {dominant_class}")
            else:
                print(f"Warning: No cls_labels found for {img_name[0]}, using default zero mask.")
                gt_mask = torch.zeros((1, inputs.shape[2], inputs.shape[3]), dtype=torch.uint8, device=device)

            for i in range(inputs.size(0)):
                base_name = os.path.basename(os.path.splitext(img_name[i])[0])
                cam_heatmap_path = os.path.join(cam_heatmap_dir, f"cam_heatmap_{base_name}_epoch_{epoch + 1}.pth")
                binary_mask_path = os.path.join(binary_mask_dir, f"binary_mask_{base_name}_epoch_{epoch + 1}.pth")
                os.makedirs(os.path.dirname(cam_heatmap_path), exist_ok=True)
                os.makedirs(os.path.dirname(binary_mask_path), exist_ok=True)
                torch.save(cam[i].cpu(), cam_heatmap_path)
                torch.save(binary_mask[i].cpu(), binary_mask_path)
                print(f"Saved CAM heatmap: {cam_heatmap_path}, Binary Mask: {binary_mask_path}")

                sample_inputs = inputs[i:i+1]
                sample_gt_mask = gt_mask[i:i+1] if gt_mask is not None else None
                sample_img_name = [img_name[i]] if img_name else None
                visualize_original_and_combined(model, sample_inputs, 1, pred_dir, feature_extractor, loss_function,
                                               fg_loss_fn, bg_loss_fn, h5_path=h5_paths[i] if h5_paths else None,
                                               epoch=epoch + 1, gt_mask=sample_gt_mask, img_name=sample_img_name)
                visualize_original_and_binary_masks(model, sample_inputs, 1, pred_dir, feature_extractor, loss_function,
                                                   fg_loss_fn, bg_loss_fn, h5_path=h5_paths[i] if h5_paths else None,
                                                   epoch=epoch + 1, gt_mask=sample_gt_mask, img_name=sample_img_name)
                visualize_original_and_combined_heatmap(model, sample_inputs, 1, pred_dir, feature_extractor, loss_function,
                                                       fg_loss_fn, bg_loss_fn, h5_path=h5_paths[i] if h5_paths else None,
                                                       epoch=epoch + 1, gt_mask=sample_gt_mask, img_name=sample_img_name)
                visualize_original_and_heatmaps(model, sample_inputs, 1, pred_dir, feature_extractor, loss_function,
                                               fg_loss_fn, bg_loss_fn, h5_path=h5_paths[i] if h5_paths else None,
                                               epoch=epoch + 1, gt_mask=sample_gt_mask, img_name=sample_img_name)
                visualize_original_and_binary_per_class(model, sample_inputs, 1, pred_dir, feature_extractor, loss_function,
                                                        fg_loss_fn, bg_loss_fn, h5_path=h5_paths[i] if h5_paths else None,
                                                        epoch=epoch + 1, gt_mask=sample_gt_mask, img_name=sample_img_name)

        # Skip val_gt_mask loading since it doesn't align with training data
        val_gt_mask = None
        print("Skipping val_gt_mask loading due to mismatch with training data. Using cls_labels for validation.")

        # all_cls_acc4, avg_cls_acc4, miou_score, fw_iou_metric, b_iou_metric, dice_score, val_cls_loss = validate(
        #     model, val_loader, cfg, loss_function, miou_metric, fw_iou_metric, b_iou_metric, dice_metric, cfg.train.epoch, H5_VAL_DIR, gt_mask=val_gt_mask
        # )
        all_cls_acc4, avg_cls_acc4, miou_score, fw_iou_metric, b_iou_metric, dice_score, val_cls_loss = validate(
            model, val_loader, cfg, loss_function, miou_metric, fw_iou_metric, b_iou_metric, dice_metric, cfg.train.epoch, H5_TRAIN_DIR,
            feature_extractor, gt_mask
        )
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([cfg.train.epoch, n_iter + 1, val_cls_loss, all_cls_acc4, miou_score.item() if miou_score.numel() > 0 else 0.0,
                            dice_score.item() if dice_score.numel() > 0 else 0.0, "", "", best_miou])
        print(f"Final Validation - Loss: {val_cls_loss:.4f}, Acc4: {all_cls_acc4:.2f}%, mIoU: {miou_score.item() if miou_score.numel() > 0 else 0.0:.2f}%, Dice: {dice_score.item() if dice_score.numel() > 0 else 0.0:.2f}%")

        if miou_score.item() > best_miou:
            best_miou = miou_score.item()
            save_path = os.path.join(ckpt_dir, f"best_stage1_epoch_{cfg.train.epoch}.pth")
            torch.save({"cfg": cfg, "epoch": cfg.train.epoch, "iter": n_iter, "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(), "best_miou": best_miou}, save_path)
            print(f"Saved best Stage 1 model with mIoU: {best_miou:.4f} at {save_path}")

    print("\nTraining completed. Check results in prediction directory.")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CAM generation and visualization pipeline.")
    parser.add_argument('--config', type=str, default=r"D:\NghienCuu\NghienCuuPaper\Source_Code\WSSS-01\src\modified_externals\ViLa-PIP\work_dirs\bcss_wsss\classification\config.yaml",
                        help="Path to the YAML configuration file.")
    parser.add_argument('--gpu', type=int, default=0, help="GPU device ID.")
    args = parser.parse_args()

    if not args.config:
        raise ValueError("No config file specified. Please provide a --config argument.")
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found at: {args.config}")
    if not args.config.endswith(('.yaml', '.yml')):
        raise TypeError(f"Config file must be a YAML file (.yaml or .yml), got: {args.config}")
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            content = f.read()
            if not content.strip():
                raise ValueError(f"Config file is empty: {args.config}")
        cfg = OmegaConf.load(args.config)
    except Exception as e:
        print(f"Warning: Failed to load config file {args.config}: {str(e)}. Exiting.")
        sys.exit(1)

    vilamil_config = OmegaConf.create()
    OmegaConf.update(vilamil_config, "text_prompt", cfg.model.text_prompt)
    vilamil_config.prototype_number = cfg.model.prototype_number
    vilamil_config.n_ratio = cfg.model.n_ratio
    vilamil_config.backbone = cfg.model.backbone
    vilamil_config.label_feature_path = cfg.model.label_feature_path
    vilamil_config.input_size = cfg.model.input_size

    if not hasattr(cfg, 'dataset'):
        cfg.dataset = OmegaConf.create()
    cfg.dataset.cls_num_classes = 4  # Changed to 4 classes
    cfg.dataset.input_size = [224, 224]
    if not hasattr(cfg, 'work_dir'):
        cfg.work_dir = OmegaConf.create()
    cfg.work_dir.dir = os.path.dirname(args.config)
    cfg.work_dir.ckpt_dir = os.path.join(cfg.work_dir.dir, "checkpoints")
    cfg.work_dir.pred_dir = os.path.join(cfg.work_dir.dir, "predictions")
    cfg.work_dir.csv_dir = os.path.join(cfg.work_dir.dir, "csv_logs")

    timestamp = "{0:%Y-%m-%d-%H-%M}".format(datetime.datetime.now())
    cfg.work_dir.ckpt_dir = os.path.join(cfg.work_dir.ckpt_dir, timestamp)
    cfg.work_dir.pred_dir = os.path.join(cfg.work_dir.pred_dir, timestamp)
    cfg.work_dir.csv_dir = os.path.join(cfg.work_dir.csv_dir, timestamp)
    os.makedirs(cfg.work_dir.dir, exist_ok=True)
    os.makedirs(cfg.work_dir.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.pred_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.csv_dir, exist_ok=True)
    print('\nArgs: %s' % args)
    print('\nConfigs: %s' % cfg)
    print('\nViLa_MIL_Model Config: %s' % vilamil_config)

    # Define the dataset and dataloader
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

    train_dataset = BCSS_WSSSTrainingDataset(transform=train_transform)
    train_subset = Subset(train_dataset, range(min(10, len(train_dataset))))  # Limit to 10 training images
    val_dataset = BCSSWSSSDataset(split='val', transform=val_transform)
    val_subset = Subset(val_dataset, range(min(10, len(val_dataset))))  # Limit to 10 validation images
    test_dataset = BCSSWSSSDataset(split='test', transform=val_transform)
    test_subset = Subset(test_dataset, range(min(10, len(test_dataset))))  # Limit to 10 test images

    def custom_collate(batch):
        img_name = [item[0] for item in batch]
        # Determine the split based on img_name
        is_training = any('training' in name.lower() or 'dataset_features_extraction' in name.lower() for name in img_name)
        is_test = any('test' in name.lower() or 'test_feature_extraction' in name.lower() for name in img_name)
        is_val = any('val' in name.lower() or 'val_feature_extraction' in name.lower() for name in img_name)
        h5_base_dir = H5_TRAIN_DIR if is_training else H5_TEST_DIR if is_test else H5_VAL_DIR if is_val else H5_TRAIN_DIR
        h5_paths = []
        cls_labels_list = []
        for name in img_name:
            base_name = os.path.basename(name)
            base_name_no_ext = os.path.splitext(base_name)[0]
            h5_filename = f"{base_name_no_ext}.h5"
            h5_path = os.path.join(h5_base_dir, h5_filename)
            h5_paths.append(h5_path)
            # Extract cls_labels from training image name (e.g., [1101])
            if is_training and '[' in base_name and ']' in base_name:
                cls_str = base_name[base_name.find('[') + 1:base_name.find(']')]
                cls_idx = int(cls_str) if cls_str.isdigit() else 0  # Default to 0 if invalid
                cls_labels_list.append(torch.tensor([1.0 if i == cls_idx else 0.0 for i in range(4)], dtype=torch.float32))
            else:
                cls_labels_list.append(torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float32))  # Default for val/test
        inputs = torch.stack([item[1] for item in batch])
        cls_labels = torch.stack(cls_labels_list)
        masks = [torch.zeros((1, inputs.size(2), inputs.size(3)), dtype=torch.long) if m is None else m.unsqueeze(0) for m in [item[3] for item in batch]]
        masks = torch.stack(masks)
        return h5_paths, inputs, cls_labels, masks

    # Create separate data loaders
    train_loader = DataLoader(train_subset, batch_size=1, shuffle=True, num_workers=0, collate_fn=custom_collate)
    val_loader = DataLoader(val_subset, batch_size=1, shuffle=False, num_workers=0, collate_fn=custom_collate)
    test_loader = DataLoader(test_subset, batch_size=1, shuffle=False, num_workers=0, collate_fn=custom_collate)
    print(f"Training samples: {len(train_subset)}, Validation samples: {len(val_subset)}, Test samples: {len(test_subset)}")

    # Update label_feature_path to include all three directories
    cfg.model.label_feature_path = [
        H5_TEST_DIR,
        H5_VAL_DIR,
        H5_TRAIN_DIR
    ]

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    set_seed(42)

    try:
        print("Initializing ViLa_MIL_Model...")
        vilamil_model = ViLa_MIL_Model(config=vilamil_config, num_classes=4)
        print(f"ViLa_MIL_Model initialized with hidden_size: {vilamil_model.D}")
        vilamil_model = vilamil_model.to(device)
        print("ViLa_MIL_Model moved to device:", device)
        print("Initializing ClsNetwork...")
        model = ClsNetwork(
            backbone=vilamil_model,
            stride=cfg.model.backbone.stride,
            cls_num_classes=4,
            n_ratio=cfg.model.n_ratio,
            pretrained=cfg.train.pretrained if hasattr(cfg.train, 'pretrained') else False,
            l_fea_path=H5_TRAIN_DIR,  # Default for model initialization
            text_prompt=vilamil_config.text_prompt
        )
        print("ClsNetwork initialized successfully")
        model = model.to(device)
        print("ClsNetwork moved to device:", device)
        print(f"Successfully loaded ClsNetwork with ViLa_MIL_Model backbone and num_classes: {model.cls_num_classes}")
    except Exception as e:
        print(f"Error during model initialization: {str(e)}")
        raise RuntimeError(f"Failed to initialize ClsNetwork with ViLa_MIL_Model: {str(e)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    loss_function = nn.BCEWithLogitsLoss().to(device)
    mask_adapter = MaskAdapter_DynamicThreshold(alpha=cfg.train.mask_adapter_alpha).to(device)
    # Initialize feature_extractor with a single default H5 path, relying on model to handle directory switching
    feature_extractor = FeatureExtractor(mask_adapter, h5_path=H5_TRAIN_DIR, clip_size=224, device=device)
    fg_loss_fn = InfoNCELossFG(temperature=0.07).to(device)
    bg_loss_fn = InfoNCELossBG(temperature=0.07).to(device)
    miou_metric = JaccardIndex(task="multiclass", num_classes=4).to(device)
    if HAS_TORCHMETRICS:
        dice_metric = Dice(num_classes=4).to(device)
    else:
        dice_metric = DiceMetric(num_classes=4, device=device)

    train(cfg, model, optimizer, feature_extractor, loss_function, fg_loss_fn, bg_loss_fn, miou_metric, dice_metric, device, train_loader, val_loader, test_loader, cfg.work_dir.ckpt_dir, cfg.work_dir.pred_dir, cfg.work_dir.csv_dir)