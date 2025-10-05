import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
import h5py
from PIL import Image
import numpy as np
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from scipy.ndimage import binary_dilation

# Import required modules from model and utils
ROOT_FOLDER = r"D:\NghienCuu\NghienCuuPaper\Source_Code\WSSS-01\src\modified_externals\ViLa-PIP"
sys.path.insert(0, ROOT_FOLDER)

from model.model_ViLa_MIL import ViLa_MIL_Model
from model.projector import PLIPProjector
from model.model import ClsNetwork
from utils.fgbg_feature import FeatureExtractor, MaskAdapter_DynamicThreshold
from utils.contrast_loss import InfoNCELossFG, InfoNCELossBG
from utils.hierarchical_utils import pair_features
from datasets.bcss_wsss import BCSS_WSSSTrainingDataset

# Define directories and constants
CHECKPOINT_PATH = r"D:\NghienCuu\NghienCuuPaper\Source_Code\WSSS-01\src\modified_externals\ViLa-PIP\work_dirs\bcss_wsss\classification\checkpoints\2025-10-06-00-00\best_stage1_epoch_3.pth"
PREDICTION_DIR = r"D:\NghienCuu\NghienCuuPaper\Source_Code\WSSS-01\src\modified_externals\ViLa-PIP\work_dirs\bcss_wsss\classification\predictions\2025-10-06-00-00\CAM_heatmap"
OUTPUT_DIR = r"E:\Result_CAM_Heatmap"
H5_TRAIN_DIR = r"D:\NghienCuu\NghienCuuPaper\Source_Code\WSSS-01\src\modified_externals\ViLa-PIP\features_extraction\dataset_features_extraction"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 4
INPUT_SIZE = [224, 224]

# Load configuration
CONFIG_PATH = r"D:\NghienCuu\NghienCuuPaper\Source_Code\WSSS-01\src\modified_externals\ViLa-PIP\work_dirs\bcss_wsss\classification\config.yaml"
cfg = OmegaConf.load(CONFIG_PATH)

# Updated palette to match BCSS dataset colors: Red, Green, Blue, Purple
palette = np.array([
    [255, 0, 0],    # TUM - Red
    [0, 255, 0],    # STR - Green
    [0, 0, 255],    # LYM - Blue
    [128, 0, 128],  # NEC - Purple
])

class_descriptions = {
    0: ("TUM", "Tumor", tuple(palette[0])),
    1: ("STR", "Stroma", tuple(palette[1])),
    2: ("LYM", "Lymphocyte", tuple(palette[2])),
    3: ("NEC", "Necrosis", tuple(palette[3])),
}

def generate_cam(model, x, coords, cls_labels, fg_bg_labels, feature_extractor, loss_function, fg_loss_fn, bg_loss_fn, h5_path=None, img_name=None, save_dir=None, epoch=None):
    model.eval()
    device = next(model.parameters()).device
    num_classes = 4

    with torch.no_grad():
        x = x.to(device).float()
        coords = coords.to(device).float() if coords is not None else None
        fg_bg_labels = fg_bg_labels.to(device).float() if fg_bg_labels is not None else torch.ones((x.shape[0], 1), dtype=torch.float32, device=device)
        
        if cls_labels is not None:
            if cls_labels.dim() == 1:
                cls_labels = F.one_hot(cls_labels.long(), num_classes=num_classes).float().to(device)
            else:
                cls_labels = cls_labels.to(device).float()
        else:
            cls_labels = torch.full((x.shape[0], num_classes), 0.25, dtype=torch.float32, device=device)

        cam1, cam2, cam3, cam4, loss = model.generate_cam(x, cls_labels, h5_path=h5_path, coords=coords, fg_bg_labels=fg_bg_labels)
        cam = torch.stack([cam1, cam2, cam3, cam4], dim=1)
        cam = torch.max(cam, dim=1)[0]

        orig_h, orig_w = x.shape[2], x.shape[3]
        if cam.shape[2:] != (orig_h, orig_w):
            cam = F.interpolate(cam, size=(orig_h, orig_w), mode='bilinear', align_corners=False)

        threshold = 0.1
        cam_norm = torch.zeros_like(cam)
        for c in range(num_classes):
            min_v = cam[:, c, :, :].min()
            max_v = cam[:, c, :, :].max()
            if max_v > min_v:
                cam_norm[:, c, :, :] = (cam[:, c, :, :] - min_v) / (max_v - min_v)
        cam_softmax = F.softmax(cam_norm, dim=1)
        max_values = torch.max(cam_softmax, dim=1, keepdim=True)[0]
        max_indices = torch.argmax(cam_softmax, dim=1, keepdim=True)
        binary_mask = F.one_hot(max_indices.squeeze(1), num_classes=num_classes).permute(0, 3, 1, 2).float()

        if feature_extractor is not None and cls_labels is not None and fg_bg_labels is not None:
            batch_info = feature_extractor.process_batch(img_name, cam, cls_labels, coords=coords, x=x)
            if batch_info is not None:
                fg_features, bg_features = batch_info['fg_features'], batch_info['bg_features']
                set_info = pair_features(fg_features, bg_features, model.l_fea, cls_labels, device=device)
                fg_features, bg_features, fg_pro, bg_pro = set_info['fg_features'], set_info['bg_features'], set_info['fg_text'], set_info['bg_text']
                fg_loss = fg_loss_fn(fg_features, fg_pro, bg_pro) if fg_loss_fn and fg_pro is not None else 0.0
                bg_loss = bg_loss_fn(bg_features, fg_pro, bg_pro) if bg_loss_fn and bg_pro is not None else 0.0
                cls1, cls2, cls3, cls4 = cam1.mean(dim=(2, 3)), cam2.mean(dim=(2, 3)), cam3.mean(dim=(2, 3)), cam4.mean(dim=(2, 3))
                Y_prob = (cls1 + cls2 + cls3 + cls4) / 4
                cls_loss = loss_function(Y_prob, cls_labels)
                loss = cls_loss + (fg_loss + bg_loss + 0.0005 * torch.mean(cam)) * 0.1
            else:
                cls1, cls2, cls3, cls4 = cam1.mean(dim=(2, 3)), cam2.mean(dim=(2, 3)), cam3.mean(dim=(2, 3)), cam4.mean(dim=(2, 3))
                Y_prob = (cls1 + cls2 + cls3 + cls4) / 4
                loss = loss_function(Y_prob, cls_labels)
        else:
            cls1, cls2, cls3, cls4 = cam1.mean(dim=(2, 3)), cam2.mean(dim=(2, 3)), cam3.mean(dim=(2, 3)), cam4.mean(dim=(2, 3))
            Y_prob = (cls1 + cls2 + cls3 + cls4) / 4
            loss = loss_function(Y_prob, cls_labels)

    return cam, binary_mask, loss

def create_combined_heatmap(cam_np: np.ndarray) -> np.ndarray:
    """Combine heatmap channels by taking the maximum value across classes."""
    if cam_np.ndim == 3 and cam_np.shape[0] == 4:  # [4, height, width]
        return np.max(cam_np, axis=0)  # Max pooling across classes
    elif cam_np.ndim == 2:  # [height, width]
        return cam_np
    else:
        raise ValueError(f"Unexpected cam_np shape: {cam_np.shape}")

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

        import matplotlib.pyplot as plt
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

        save_path = os.path.join(save_dir, f'original_combined_heatmap_{os.path.basename(os.path.splitext(img_name[0])[0])}_epoch_{epoch}.png') if epoch else os.path.join(save_dir, f'original_combined_heatmap_{os.path.basename(os.path.splitext(img_name[0])[0])}.png')
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

        import matplotlib.pyplot as plt
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

        save_path = os.path.join(save_dir, f'original_heatmaps_{os.path.basename(os.path.splitext(img_name[0])[0])}_epoch_{epoch}.png') if epoch else os.path.join(save_dir, f'original_heatmaps_{os.path.basename(os.path.splitext(img_name[0])[0])}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Saved original and heatmaps: {save_path}")

def infer_and_save_cam_heatmaps(model, feature_extractor, loss_function, fg_loss_fn, bg_loss_fn, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize dataset to load images
    train_transform = A.Compose([
        A.Resize(height=INPUT_SIZE[0], width=INPUT_SIZE[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    dataset = BCSS_WSSSTrainingDataset(transform=train_transform)
    
    # Precompute dataset image names for faster lookup
    dataset_names = {os.path.basename(os.path.splitext(name)[0]): i for i, (name, _, _, _) in enumerate(dataset)}
    print(f"Loaded {len(dataset_names)} images from dataset. Available names: {list(dataset_names.keys())[:5]}...")  # Debug dataset names

    pth_files = [f for f in os.listdir(PREDICTION_DIR) if f.endswith('.pth')]
    if not pth_files:
        print(f"No .pth files found in {PREDICTION_DIR}. Exiting.")
        return

    model.eval()
    with torch.no_grad():
        for pth_file in tqdm(pth_files, desc="Generating CAM Heatmaps", total=len(pth_files)):
            pth_path = os.path.join(PREDICTION_DIR, pth_file)
            # Extract base_name by removing 'cam_heatmap_' and everything after '_epoch_X'
            base_name = pth_file.replace('cam_heatmap_', '').split('_epoch_')[0]
            print(f"Processing {pth_file}, base_name: {base_name}")
            
            # Load CAM heatmap from .pth file
            cam_heatmap = torch.load(pth_path, map_location=DEVICE)
            if cam_heatmap.ndim == 4 and cam_heatmap.shape[1] == NUM_CLASSES:
                cam_heatmap = cam_heatmap.squeeze(0)  # Remove batch dimension if present
            elif cam_heatmap.ndim == 3 and cam_heatmap.shape[0] == 1:
                cam_heatmap = cam_heatmap.squeeze(0)
            elif cam_heatmap.ndim != 3 or cam_heatmap.shape[0] != NUM_CLASSES:
                print(f"Unexpected cam_heatmap shape {cam_heatmap.shape} in {pth_file}, skipping.")
                continue

            # Move cam_heatmap to CPU for processing
            cam_heatmap = cam_heatmap.cpu()

            # Find matching image in dataset
            img_name = None
            idx = dataset_names.get(base_name)
            if idx is not None:
                name, x, cls_labels, _ = dataset[idx]
                img_name = name
                x = x.unsqueeze(0)  # Add batch dimension
            else:
                print(f"No matching image found for {base_name} in dataset, skipping.")
                continue

            # Generate CAM (optional, for consistency with visualization)
            epoch = pth_file.split('_epoch_')[-1].replace('.pth', '') if 'epoch' in pth_file else None
            cls_labels = cls_labels.unsqueeze(0).to(DEVICE) if cls_labels is not None else torch.full((1, NUM_CLASSES), 0.25, dtype=torch.float32, device=DEVICE)
            fg_bg_labels = torch.ones((1, 1), dtype=torch.float32).to(DEVICE)
            print(f"Generating CAM for {img_name}")
            cam, _, _ = generate_cam(
                model, x, None, cls_labels, fg_bg_labels,
                feature_extractor=feature_extractor, loss_function=loss_function,
                fg_loss_fn=fg_loss_fn, bg_loss_fn=bg_loss_fn,
                h5_path=os.path.join(H5_TRAIN_DIR, f"{base_name}.h5"), img_name=[img_name], save_dir=None, epoch=epoch
            )

            # Save combined heatmap
            combined_heatmap = create_combined_heatmap(cam_heatmap.numpy())
            output_path = os.path.join(output_dir, f"combined_heatmap_{base_name}_epoch_{epoch}.png")
            combined_heatmap_pil = Image.fromarray((combined_heatmap * 255).astype(np.uint8))  # Scale to 0-255 for PNG
            combined_heatmap_pil.save(output_path)
            print(f"Saved combined heatmap: {output_path}")

            # Copy .pth file to output directory
            torch.save(cam_heatmap.cpu(), os.path.join(output_dir, pth_file.replace(PREDICTION_DIR, '').lstrip('/\\')))
            print(f"Copied .pth file: {pth_file} to {output_path.replace('.png', '.pth')}")

            # Generate visualizations
            gt_mask = cls_labels[0].argmax().item() * torch.ones((INPUT_SIZE[0], INPUT_SIZE[1]), dtype=torch.uint8).to(DEVICE)
            visualize_original_and_combined_heatmap(model, x, 1, output_dir, feature_extractor, loss_function, fg_loss_fn, bg_loss_fn,
                                                 h5_path=os.path.join(H5_TRAIN_DIR, f"{base_name}.h5"), epoch=epoch, gt_mask=gt_mask.unsqueeze(0), img_name=[img_name])
            visualize_original_and_heatmaps(model, x, 1, output_dir, feature_extractor, loss_function, fg_loss_fn, bg_loss_fn,
                                         h5_path=os.path.join(H5_TRAIN_DIR, f"{base_name}.h5"), epoch=epoch, gt_mask=gt_mask.unsqueeze(0), img_name=[img_name])

def main():
    vilamil_config = OmegaConf.create()
    OmegaConf.update(vilamil_config, "text_prompt", cfg.model.text_prompt)
    vilamil_config.prototype_number = cfg.model.prototype_number
    vilamil_config.n_ratio = cfg.model.n_ratio
    vilamil_config.backbone = cfg.model.backbone
    vilamil_config.label_feature_path = cfg.model.label_feature_path
    vilamil_config.input_size = cfg.model.input_size

    vilamil_model = ViLa_MIL_Model(config=vilamil_config, num_classes=NUM_CLASSES)
    vilamil_model = vilamil_model.to(DEVICE)
    
    model = ClsNetwork(
        backbone=vilamil_model,
        stride=cfg.model.backbone.stride,
        cls_num_classes=NUM_CLASSES,
        n_ratio=cfg.model.n_ratio,
        pretrained=False,
        l_fea_path=H5_TRAIN_DIR,
        text_prompt=vilamil_config.text_prompt
    )
    model = model.to(DEVICE)
    
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    print(f"Loaded checkpoint from {CHECKPOINT_PATH} at epoch {checkpoint['epoch']} with best mIoU {checkpoint['best_miou']:.4f}")

    loss_function = nn.BCEWithLogitsLoss().to(DEVICE)
    mask_adapter = MaskAdapter_DynamicThreshold(alpha=cfg.train.mask_adapter_alpha).to(DEVICE)
    feature_extractor = FeatureExtractor(mask_adapter, h5_path=H5_TRAIN_DIR, clip_size=224, device=DEVICE)
    fg_loss_fn = InfoNCELossFG(temperature=0.07).to(DEVICE)
    bg_loss_fn = InfoNCELossBG(temperature=0.07).to(DEVICE)

    infer_and_save_cam_heatmaps(model, feature_extractor, loss_function, fg_loss_fn, bg_loss_fn, OUTPUT_DIR)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    main()
    