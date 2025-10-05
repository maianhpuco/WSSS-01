"""
CAM generation and visualization utilities
"""

import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import ttach as tta
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader

from .evaluate import ConfusionMatrixAllClass
from .hierarchical_utils import merge_to_parent_predictions, merge_subclass_cams_to_parent
from .pyutils import AverageMeter

def get_seg_label(cams, inputs, label):
    """Generate segmentation labels from CAMs"""
    with torch.no_grad():
        b, c, h, w = inputs.shape
        label = label.view(b, -1, 1, 1).cpu().data.numpy()
        cams = cams.cpu().data.numpy()
        cams = np.maximum(cams, 0)
        
        channel_max = np.max(cams, axis=(2, 3), keepdims=True)
        channel_min = np.min(cams, axis=(2, 3), keepdims=True)
        cams = (cams - channel_min) / (channel_max - channel_min + 1e-6)
        cams = cams * label
        cams = torch.from_numpy(cams).float()
        
        cams = F.interpolate(cams, size=(h, w), mode="bilinear", align_corners=True)
        cam_max = torch.max(cams, dim=1, keepdim=True)[0]
        bg_cam = (1 - cam_max) ** 10
        cam_all = torch.cat([cams, bg_cam], dim=1)

        expected_channels = 5
        if cam_all.shape[1] != expected_channels:
            raise ValueError(f"CAM channels mismatch: got {cam_all.shape[1]}, expected {expected_channels}")

    return cam_all

def validate(model=None, data_loader=None, cfg=None, cls_loss_func=None):
    """Validation function with test-time augmentation"""
    model.eval()
    avg_meter = AverageMeter()
    fuse234_matrix = ConfusionMatrixAllClass(num_classes=5)
    
    tta_transform = tta.Compose([
        tta.HorizontalFlip(),
        tta.Multiply(factors=[0.9, 1.0, 1.1])
    ])
    with torch.no_grad():
        for data in tqdm(data_loader, total=len(data_loader), ncols=100, ascii=" >="):
            name, inputs, cls_label, labels = data
            inputs = inputs.to("cpu").float()
            labels = labels.to("cpu")
            cls_label = cls_label.to("cpu").float()

            outputs = model(inputs)
            cls1, cam1, cls2, cam2, cls3, cam3, cls4, cam4 = outputs

            cls1 = merge_to_parent_predictions(cls1, None, method=cfg.train.merge_test)
            cls2 = merge_to_parent_predictions(cls2, None, method=cfg.train.merge_test)
            cls3 = merge_to_parent_predictions(cls3, None, method=cfg.train.merge_test)
            cls4 = merge_to_parent_predictions(cls4, None, method=cfg.train.merge_test)

            cls_loss1 = cls_loss_func(cls1, cls_label)
            cls_loss2 = cls_loss_func(cls2, cls_label)
            cls_loss3 = cls_loss_func(cls3, cls_label)
            cls_loss4 = cls_loss_func(cls4, cls_label)
            cls_loss = cfg.train.l1 * cls_loss1 + cfg.train.l2 * cls_loss2 + cfg.train.l3 * cls_loss3 + cfg.train.l4 * cls_loss4

            cls4_pred = (torch.sigmoid(cls4) > 0.5).float()
            if cls4_pred.shape[1] != cls_label.shape[1]:
                raise ValueError(f"Class dimension mismatch: cls4_pred={cls4_pred.shape[1]}, cls_label={cls_label.shape[1]}")
            
            all_cls_acc4 = (cls4_pred == cls_label).all(dim=1).float().sum() / cls4_pred.shape[0] * 100
            avg_cls_acc4 = ((cls4_pred == cls_label).sum(dim=0) / cls4_pred.shape[0]).mean() * 100
            avg_meter.add({"all_cls_acc4": all_cls_acc4, "avg_cls_acc4": avg_cls_acc4, "cls_loss": cls_loss})
            
            cams1 = []
            cams2 = []
            cams3 = []
            cams4 = []
            for tta_trans in tta_transform:
                augmented_tensor = tta_trans.augment_image(inputs)
                outputs = model(augmented_tensor)
                cls1, cam1, cls2, cam2, cls3, cam3, cls4, cam4 = outputs

                cam1 = merge_subclass_cams_to_parent(cam1, None, method=cfg.train.merge_test)
                cam2 = merge_subclass_cams_to_parent(cam2, None, method=cfg.train.merge_test)
                cam3 = merge_subclass_cams_to_parent(cam3, None, method=cfg.train.merge_test)
                cam4 = merge_subclass_cams_to_parent(cam4, None, method=cfg.train.merge_test)

                cam1 = get_seg_label(cam1, augmented_tensor, cls_label).to("cpu")
                cam1 = tta_trans.deaugment_mask(cam1).unsqueeze(dim=0)
                cams1.append(cam1)
                cam2 = get_seg_label(cam2, augmented_tensor, cls_label).to("cpu")
                cam2 = tta_trans.deaugment_mask(cam2).unsqueeze(dim=0)
                cams2.append(cam2)
                cam3 = get_seg_label(cam3, augmented_tensor, cls_label).to("cpu")
                cam3 = tta_trans.deaugment_mask(cam3).unsqueeze(dim=0)
                cams3.append(cam3)
                cam4 = get_seg_label(cam4, augmented_tensor, cls_label).to("cpu")
                cam4 = tta_trans.deaugment_mask(cam4).unsqueeze(dim=0)
                cams4.append(cam4)

            cams1 = torch.cat(cams1, dim=0).mean(dim=0)
            cams2 = torch.cat(cams2, dim=0).mean(dim=0)
            cams3 = torch.cat(cams3, dim=0).mean(dim=0)
            cams4 = torch.cat(cams4, dim=0).mean(dim=0)

            fuse234 = 0.3 * cams2 + 0.3 * cams3 + 0.4 * cams4
            fuse_label234 = torch.argmax(fuse234, dim=1).to("cpu")
            
            fuse234_matrix.update(labels.detach().clone(), fuse_label234.clone())

    all_cls_acc4, avg_cls_acc4, cls_loss = avg_meter.pop('all_cls_acc4'), avg_meter.pop("avg_cls_acc4"), avg_meter.pop("cls_loss")
    fuse234_score = fuse234_matrix.compute()[2]
    model.train()
    return all_cls_acc4, avg_cls_acc4, fuse234_score, cls_loss

def generate_cam(model=None, data_loader=None, cfg=None, cls_loss_func=None):
    """Generate CAMs and save as images"""
    model.eval()
    input_size = cfg.dataset.input_size
    with torch.no_grad():
        for data in tqdm(data_loader, total=len(data_loader), ncols=100, ascii=" >="):
            name, inputs, cls_label, labels = data
            inputs = inputs.to("cpu").float()
            b, c, h, w = inputs.shape
            labels = labels.to("cpu")
            cls_label = cls_label.to("cpu").float()

            outputs = model(inputs)
            cls1, cam1, cls2, cam2, cls3, cam3, cls4, cam4 = outputs

            cam1 = merge_subclass_cams_to_parent(cam1, None, method=cfg.train.merge_test)
            cam2 = merge_subclass_cams_to_parent(cam2, None, method=cfg.train.merge_test)
            cam3 = merge_subclass_cams_to_parent(cam3, None, method=cfg.train.merge_test)
            cam4 = merge_subclass_cams_to_parent(cam4, None, method=cfg.train.merge_test)

            cam1 = get_seg_label(cam1, inputs, cls_label).to("cpu")
            cam2 = get_seg_label(cam2, inputs, cls_label).to("cpu")
            cam3 = get_seg_label(cam3, inputs, cls_label).to("cpu")
            cam4 = get_seg_label(cam4, inputs, cls_label).to("cpu")

            fuse234 = 0.3 * cam2 + 0.3 * cam3 + 0.4 * cam4
            output_fuse234 = torch.argmax(fuse234, dim=1).long()

            PALETTE = [
                [255, 0, 0],
                [0, 255, 0],
                [0, 0, 255],
                [153, 0, 255],
                [255, 255, 255]
            ]

            for i in range(len(output_fuse234)):
                pred_mask = Image.fromarray(output_fuse234[i].cpu().clone().squeeze().numpy().astype(np.uint8)).convert('P')
                flat_palette = [val for sublist in PALETTE for val in sublist]
                pred_mask.putpalette(flat_palette)
                pred_mask.save(os.path.join(cfg.work_dir.pred_dir, name[i] + ".png"))
    model.train()
    return
