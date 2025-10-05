import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import h5py
import psutil

# Import PLIPProjector from model/projector.py
import sys
import os
ROOT_FOLDER = r"D:\NghienCuu\NghienCuuPaper\Source_Code\WSSS-01\src\modified_externals\ViLa-PIP"
sys.path.insert(0, ROOT_FOLDER)

from model.projector import PLIPProjector

class MaskAdapter_DynamicThreshold(nn.Module):
    def __init__(self, alpha, mask_cam=False):
        super(MaskAdapter_DynamicThreshold, self).__init__()
        self.alpha = alpha
        self.mask_cam = mask_cam

    def forward(self, x):
        binary_mask = []
        for i in range(x.shape[0]):
            th = torch.max(x[i]) * self.alpha
            binary_mask.append(
                torch.where(x[i] >= th, torch.ones_like(x[i]), torch.zeros_like(x[i]))
            )
        binary_mask = torch.stack(binary_mask, dim=0)
        if self.mask_cam:
            return x * binary_mask
        else:
            return binary_mask

class FeatureExtractor(nn.Module):
    def __init__(self, mask_adapter, h5_path, clip_size=224, device=None):
        super(FeatureExtractor, self).__init__()
        self.device = device if isinstance(device, torch.device) else torch.device('cpu')
        self.mask_adapter = mask_adapter.to(self.device)
        self.h5_path = h5_path  # Restore h5_path parameter
        self.clip_size = clip_size
        self.projector = PLIPProjector(
            local_model_path=r"D:\NghienCuu\NghienCuuPaper\Source_Code\WSSS-01\pretrained_models\vinid_plip",
            device=self.device
        ).to(self.device)
        self.feature_projector = nn.Linear(1024, 512).to(self.device)
        self.coord_projection = nn.Linear(2, 128).to(self.device)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
        ).to(self.device)
        print(f"FeatureExtractor initialized - mask_adapter device: {self.mask_adapter[0].weight.device if isinstance(self.mask_adapter, nn.ModuleList) else 'N/A'}")
        print(f"FeatureExtractor initialized - projector device: {next(self.projector.parameters()).device}")
        print(f"FeatureExtractor initialized - feature_projector device: {self.feature_projector.weight.device}")
        print(f"FeatureExtractor initialized - coord_projection device: {self.coord_projection.weight.device}")
        print(f"FeatureExtractor initialized - feature_extractor device: {self.feature_extractor[0].weight.device}")
        
    @torch.no_grad()
    def extract_features(self, img_name, cam_224, cam_224_mask, x, label, device, coords=None):
        if x is not None and x.device != self.device:
            raise RuntimeError(f"Device mismatch: input tensor is on {x.device}, but FeatureExtractor is on {self.device}")
        if coords is not None and coords.device != self.device:
            raise RuntimeError(f"Device mismatch: coords is on {coords.device}, but FeatureExtractor is on {self.device}")

        batch_size = label.shape[0]
        num_classes = label.shape[1]
        
        if label.dim() != 2 or label.shape != (batch_size, num_classes):
            raise ValueError(f"Expected label shape [{batch_size}, {num_classes}], got {label.shape}")

        if cam_224.shape != (batch_size, num_classes, self.clip_size, self.clip_size):
            raise ValueError(f"Expected cam_224 shape [{batch_size}, {num_classes}, {self.clip_size}, {self.clip_size}], got {cam_224.shape}")
        if cam_224_mask.shape != (batch_size, num_classes, self.clip_size, self.clip_size):
            raise ValueError(f"Expected cam_224_mask shape [{batch_size}, {num_classes}, {self.clip_size}, {self.clip_size}], got {cam_224_mask.shape}")

        fg_features = []
        bg_features = []
        for i in range(batch_size):
            slide_id = img_name[i] if isinstance(img_name, (list, tuple)) and i < len(img_name) else f"sample_{i}"
            h5_file = os.path.join(self.h5_path, f"{slide_id}.h5")

            x_i = x[i:i+1].to(self.device) if x is not None else None
            if os.path.exists(h5_file):
                with h5py.File(h5_file, 'r') as f:
                    features = torch.from_numpy(f['features'][:]).float().to(self.device)
                    print(f"H5 file {h5_file} shape: {features.shape}")
                    features = self.projector.ImageMLP(self.feature_projector(features))  # [batch_size, 512]
            else:
                if x_i is not None:
                    if x_i.dim() != 4 or x_i.shape[1] != 3 or x_i.shape[2:] != (self.clip_size, self.clip_size):
                        raise ValueError(f"Expected x_i shape [1, 3, {self.clip_size}, {self.clip_size}], got {x_i.shape}")
                    print(f"extract_features - x_i device: {x_i.device}, feature_extractor device: {self.feature_extractor[0].weight.device}")
                    features = self.feature_extractor(x_i)  # [1, 512, height/8, width/8]
                    features = F.interpolate(features, size=(self.clip_size, self.clip_size), mode='bilinear', align_corners=False)  # [1, 512, 224, 224]
                    # Use 1x1 convolution for channel reduction instead of linear projection
                    conv_1x1 = nn.Conv2d(512, 128, kernel_size=1, bias=True).to(self.device)  # Temporary conv layer
                    features = conv_1x1(features)  # [1, 128, 224, 224]
                else:
                    features = torch.randn(1, 128, self.clip_size, self.clip_size, device=self.device)

            if coords is not None and coords.shape[0] == batch_size and coords.shape[1] == 2:
                coords_i = coords[i:i+1]
                print(f"extract_features - coords_i device: {coords_i.device}, coord_projection device: {self.coord_projection.weight.device}")
                coords_projected = self.coord_projection(coords_i)  # [1, 128]
                coords_projected = coords_projected.view(1, 128, 1, 1).expand(1, 128, self.clip_size, self.clip_size)  # [1, 128, 224, 224]
                # Ensure features has the same shape as coords_projected
                if features.shape != coords_projected.shape:
                    features = F.interpolate(features, size=(self.clip_size, self.clip_size), mode='bilinear', align_corners=False) if features.shape[2:] != (self.clip_size, self.clip_size) else features
                    features = features.view(1, 128, self.clip_size, self.clip_size)  # Ensure [1, 128, 224, 224]
                features = features + coords_projected * 0.01
                print(f"extract_features - coords shape: {coords_i.shape}, coords_projected shape: {coords_projected.shape}, features shape: {features.shape}")

            for j in range(num_classes):
                if label[i, j] > 0.01:
                    cam_mask_j = cam_224_mask[i, j].squeeze(0)  # [224, 224]
                    cam_224_j = cam_224[i, j]  # [224, 224]
                    fg_weight = cam_mask_j * cam_224_j
                    bg_weight = (1 - cam_mask_j) * (1 - cam_224_j)

                    fg_feature_j = (fg_weight.unsqueeze(0).unsqueeze(0) * features).sum(dim=[2, 3]) / (fg_weight.sum() + 1e-8)
                    bg_feature_j = (bg_weight.unsqueeze(0).unsqueeze(0) * features).sum(dim=[2, 3]) / (bg_weight.sum() + 1e-8)

                    fg_features.append(fg_feature_j)
                    bg_features.append(bg_feature_j)

        fg_features = torch.stack(fg_features) if fg_features else torch.zeros(0, features.shape[1], device=self.device)
        bg_features = torch.stack(bg_features) if bg_features else torch.zeros(0, features.shape[1], device=self.device)
        print(f"extract_features - fg_features shape: {fg_features.shape}, bg_features shape: {bg_features.shape}")
        return fg_features, bg_features, None, None
    
    def prepare_cam_mask(self, cam, N, num_classes):
        if cam.dim() != 4:
            raise ValueError(f"Expected cam to have 4 dimensions [batch_size, num_classes, h, w], got shape {cam.shape}")
        
        cam_224 = F.interpolate(
            cam,
            size=(self.clip_size, self.clip_size),
            mode="bilinear",
            align_corners=True
        )
        
        N_actual = cam_224.shape[0]
        num_classes_actual = cam_224.shape[1]
        
        expected_elements = N_actual * num_classes_actual * self.clip_size * self.clip_size
        if cam_224.numel() != expected_elements:
            raise ValueError(f"Number of elements {cam_224.numel()} does not match expected {expected_elements} for shape [{N_actual}, {num_classes_actual}, {self.clip_size}, {self.clip_size}]")
        
        cam_224_reshaped = cam_224.view(N_actual * num_classes_actual, 1, self.clip_size, self.clip_size)
        cam_224_mask = self.mask_adapter(cam_224_reshaped)
        cam_224_mask = cam_224_mask.view(N_actual, num_classes_actual, self.clip_size, self.clip_size)
        
        print(f"prepare_cam_mask - cam_224 shape: {cam_224.shape}, cam_224_mask shape: {cam_224_mask.shape}")
        return cam_224, cam_224_mask

    def prepare_image(self, img):
        img_224 = F.interpolate(
            img, (self.clip_size, self.clip_size), mode="bilinear", align_corners=True
        )
        print(f"prepare_image - img_224 shape: {img_224.shape}")
        return img_224

    @torch.no_grad()
    def get_masked_features(self, fg_features, bg_features, fg_masks, bg_masks, device):
        if fg_features.numel() > 0:
            fg_min = fg_features.min(dim=0, keepdim=True)[0]
            fg_max = fg_features.max(dim=0, keepdim=True)[0]
            normalized_fg_features = (fg_features - fg_min) / (fg_max - fg_min + 1e-8)
        else:
            normalized_fg_features = fg_features

        if bg_features.numel() > 0:
            bg_min = bg_features.min(dim=0, keepdim=True)[0]
            bg_max = bg_features.max(dim=0, keepdim=True)[0]
            normalized_bg_features = (bg_features - bg_min) / (bg_max - bg_min + 1e-8)
        else:
            normalized_bg_features = bg_features

        print(f"get_masked_features - normalized_fg_features shape: {normalized_fg_features.shape}, normalized_bg_features shape: {normalized_bg_features.shape}")
        return normalized_fg_features, normalized_bg_features

    @torch.no_grad()
    def process_batch(self, img_name, cam, label, coords=None, csv_dir=None, epoch=None, n_iter=None, device=None, x=None):
        if not torch.any(label > 0.01):
            print("No significant foreground samples in batch")
            return None

        N = label.shape[0]
        num_classes = label.shape[1]
        cam_224, cam_224_mask = self.prepare_cam_mask(cam, N, num_classes)

        fg_features, bg_features, _, _ = self.extract_features(img_name, cam_224, cam_224_mask, x, label, device, coords=coords)
        if fg_features.numel() == 0 and bg_features.numel() == 0:
            print("No valid features extracted")
            return None

        fg_features, bg_features = self.get_masked_features(fg_features, bg_features, None, None, device)
        print(f"process_batch - returned fg_features shape: {fg_features.shape}, bg_features shape: {bg_features.shape}")

        return {
            'fg_features': fg_features,
            'bg_features': bg_features,
            'cam_224': cam_224,
            'cam_224_mask': cam_224_mask
        }

    def print_debug_info(self, batch_info):
        if batch_info is None:
            print("No foreground samples in batch")
            return

        print("\nFeature extraction debug info:")
        print(f"Number of foreground samples: {batch_info['fg_features'].shape[0]}")
        print(f"Foreground features shape: {batch_info['fg_features'].shape}")
        print(f"Background features shape: {batch_info['bg_features'].shape}")
        print(f"CAM shape: {batch_info['cam_224'].shape}")
        print(f"Mask shape: {batch_info['cam_224_mask'].shape}")

    def forward(self, inputs, img_name=None, cam=None, label=None, coords=None):
        """
        Forward pass for FeatureExtractor, extracting features from inputs.
        If cam and label are not provided, uses a simplified feature extraction pipeline.
        """
        device = inputs.device
        batch_size = inputs.shape[0]
        orig_h, orig_w = inputs.shape[2], inputs.shape[3]

        # Prepare image
        img_224 = self.prepare_image(inputs)

        # If cam is not provided, generate dummy values (label is optional)
        if cam is None:
            print("Warning: cam not provided, using dummy values for feature extraction.")
            num_classes = 4  # Default number of classes
            cam = torch.randn(batch_size, num_classes, self.clip_size, self.clip_size, device=device)
            if label is None:
                label = torch.zeros(batch_size, num_classes, device=device)
                label[:, 0] = 1.0  # Dummy label for the first class
        else:
            num_classes = cam.shape[1]  # Derive num_classes from cam shape

        # Prepare CAM mask
        cam_224, cam_224_mask = self.prepare_cam_mask(cam, batch_size, num_classes)

        # Extract features using the CNN backbone
        features = self.feature_extractor(inputs)  # [batch_size, 512, height/8, width/8] due to pooling
        features = F.interpolate(features, size=(orig_h, orig_w), mode='bilinear', align_corners=False)  # [batch_size, 512, orig_h, orig_w]

        # Replace linear projection with 1x1 convolution for channel reduction
        # Replace feature_projector_to_128 (Linear) with a 1x1 Conv2d
        conv_1x1 = nn.Conv2d(512, 128, kernel_size=1, bias=True).to(device)  # Add this in __init__ if persistent
        features = conv_1x1(features)  # [batch_size, 128, orig_h, orig_w]

        print(f"forward - features shape: {features.shape}")
        return features