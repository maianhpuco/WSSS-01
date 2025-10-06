from __future__ import print_function, division
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms, models
import torch.multiprocessing
import h5py
from tqdm import tqdm
from PIL import Image
import argparse
import time
import re
import logging

torch.multiprocessing.set_sharing_strategy('file_system')
# Import
import sys
import os
# Add root directory to Python path
ROOT_FOLDER = r"D:\NghienCuu\NghienCuuPaper\Source_Code\WSSS-01\src\modified_models\ViLa-PIP"
sys.path.insert(0, ROOT_FOLDER)
# Import from bcss_wsss.py
from datasets.bcss_wsss import BCSS_WSSSTrainingDataset, BCSS_WSSS_TestDataset, BCSSWSSSDataset
# Base dir for BCSS-WSSS dataset (uncompressed directory)
BASE_DIR = r"D:\NghienCuu\NghienCuuPaper\Source_Code\data\data_BCSS-WSSS\BCSS-WSSS"
# Suppress dataset loading logs
logging.getLogger().setLevel(logging.WARNING)

def eval_transforms(pretrained=False):
    if pretrained:
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    else:
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    transforms_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return transforms_val

def eval_transforms_clip(pretrained=False):
    if pretrained:
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    else:
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    transforms_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=mean, std=std)
    ])
    return transforms_val

class PatchesDataset(Dataset):
    def __init__(self, slide_ids, transform=None):
        self.slide_ids = slide_ids
        self.transform = transform
        # Initialize BCSS datasets with progress bar
        print("\n" + "="*80)
        print("PROCESS PREPROCESSING UPLOAD DATASET")
        print("="*80)
        self.train_dataset = BCSS_WSSSTrainingDataset(transform=None)
        self.test_dataset = BCSS_WSSS_TestDataset(split="test", transform=None)
        self.val_dataset = BCSS_WSSS_TestDataset(split="val", transform=None)
        self.wsss_val_dataset = BCSSWSSSDataset(mask_name="val/mask", transform=None)
        self.wsss_test_dataset = BCSSWSSSDataset(mask_name="test/mask", transform=None)
        # Create a mapping from slide_id to dataset, index, label, and mask path
        self.slide_id_to_dataset = {}
        datasets = [
            (self.train_dataset, "training", None),
            (self.test_dataset, "test", self.wsss_test_dataset),
            (self.val_dataset, "val", self.wsss_val_dataset)
        ]
        for dataset, name, mask_dataset in tqdm(datasets, desc="Mapping slide IDs to datasets", unit="dataset"):
            base_mask_path = os.path.join(BASE_DIR, name, "mask") if mask_dataset else None
            for i, (img_path, _, cls_label, _) in enumerate(dataset):
                slide_id = os.path.splitext(os.path.basename(img_path))[0]
                if slide_id in self.slide_ids:
                    mask_path = os.path.join(base_mask_path, f"{slide_id}.png") if base_mask_path and os.path.exists(os.path.join(base_mask_path, f"{slide_id}.png")) else None
                    self.slide_id_to_dataset[slide_id] = (name, i, cls_label, mask_path)
        # Coords default to [0, 0] since patches are 224x224
        self.coords = [[0, 0] for _ in self.slide_ids]
        # Extract MPP from slide_ids
        self.mpps = []
        for slide_id in self.slide_ids:
            mpp_match = re.search(r'MPP-(\d+\.\d+)', slide_id)
            mpp = float(mpp_match.group(1)) if mpp_match else 0.2500
            self.mpps.append(mpp)

    def __getitem__(self, index):
        slide_id = self.slide_ids[index]
        if slide_id not in self.slide_id_to_dataset:
            raise ValueError(f"Slide ID {slide_id} not found in BCSS-WSSS datasets")
        dataset_name, dataset_idx, cls_label, mask_path = self.slide_id_to_dataset[slide_id]
        # Load image based on dataset
        if dataset_name == "training":
            img_path, _, _, _ = self.train_dataset[dataset_idx]
            full_path = os.path.join(BASE_DIR, "training", img_path)
            mask = None
        elif dataset_name == "test":
            img_path, _, _, mask_data = self.test_dataset[dataset_idx]
            full_path = os.path.join(BASE_DIR, "test", "img", img_path)
            mask = mask_data if mask_data is not None else torch.zeros((224, 224), dtype=torch.float32)
        elif dataset_name == "val":
            img_path, _, _, mask_data = self.val_dataset[dataset_idx]
            full_path = os.path.join(BASE_DIR, "val", "img", img_path)
            mask = mask_data if mask_data is not None else torch.zeros((224, 224), dtype=torch.float32)
        # Load image
        try:
            img = Image.open(full_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {full_path}: {e}")
            raise
        if self.transform is not None:
            img = self.transform(img)
            # Apply transform to mask only if it exists and is a tensor
            if mask is not None and not isinstance(mask, torch.Tensor):
                mask = transforms.ToTensor()(mask)
            elif mask is None:
                mask = torch.zeros((1, 224, 224), dtype=torch.float32)  # Default mask as zero tensor
        coord = torch.tensor(self.coords[index], dtype=torch.float32)
        mpp = self.mpps[index]
        return img, mask, coord, slide_id, cls_label, mpp

    def __len__(self):
        return len(self.slide_ids)

class ResNet50_Trunc_1024(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet50_Trunc_1024, self).__init__()
        # Load pre-trained ResNet-50
        resnet = models.resnet50(pretrained=pretrained)
        # Remove the final fully connected layer and global average pooling
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        # Interpolate to 14x14 feature map to match RN50 (stride 16)
        self.interpolate = nn.Upsample(size=(14, 14), mode='bilinear', align_corners=False)
        # Projection layer to reduce to 1024 dimensions
        self.projection = nn.Linear(2048, 1024)
        
    def forward(self, x):
        # x: [batch_size, 3, 224, 224]
        print(f"ResNet50 input shape: {x.shape}")
        features = self.features(x)  # [batch_size, 2048, 7, 7]
        print(f"ResNet50 features shape before interpolate: {features.shape}")
        # Interpolate to 14x14 to match RN50 patch grid
        features = self.interpolate(features)  # [batch_size, 2048, 14, 14]
        print(f"ResNet50 features shape after interpolate: {features.shape}")
        # Reshape to [batch_size, 2048, 196]
        features = features.view(features.size(0), features.size(1), -1)  # [batch_size, 2048, 196]
        # Transpose to [batch_size, 196, 2048]
        features = features.permute(0, 2, 1)  # [batch_size, 196, 2048]
        # Project to 1024 dimensions
        features = self.projection(features)  # [batch_size, 196, 1024]
        # Add CLS-like token (average of patch features)
        cls_token = features.mean(dim=1, keepdim=True)  # [batch_size, 1, 1024]
        features = torch.cat([cls_token, features], dim=1)  # [batch_size, 197, 1024]
        print(f"ResNet50 output shape: {features.shape}")
        return features

def resnet50_trunc_1024(pretrained=True):
    """Constructs a Modified ResNet-50 model with 1024-dim output and 197 patches."""
    model = ResNet50_Trunc_1024(pretrained=pretrained)
    return model

def save_embeddings(model, fname, batch, mask, coord, cls_label, mpp, enc_name, overwrite=False):
    if os.path.isfile(f'{fname}.h5') and not overwrite:
        print(f"Skipping existing file: {fname}.h5")
        return None
    batch = batch.to(torch.device('cuda'))
    start_time = time.time()
    with torch.no_grad():
        try:
            embedding = model(batch).detach().cpu().numpy()  # [batch_size, 197, 1024]
            embedding = embedding.squeeze(0)  # [197, 1024] for batch_size=1
        except Exception as e:
            print(f"Error extracting features for {fname}: {e}")
            raise
    end_time = time.time()
    print(f"Feature extraction for {fname} took {end_time - start_time:.2f} seconds")
    coord = coord.cpu().numpy()
    cls_label = cls_label.cpu().numpy()
    mpp = float(mpp)
    mask_data = mask.cpu().numpy().squeeze() if mask is not None else np.zeros((224, 224), dtype=np.float32)
    # Ensure embedding is 2D [197, 1024]
    if embedding.ndim != 2 or embedding.shape[0] != 197 or embedding.shape[1] != 1024:
        raise ValueError(f"Embedding shape {embedding.shape} is invalid, expected [197, 1024]")
    try:
        h5f = h5py.File(f'{fname}.h5', 'w')
        h5f.create_dataset('features', data=embedding)
        h5f.create_dataset('coords', data=coord)
        h5f.create_dataset('mask', data=mask_data)
        h5f.attrs['label'] = cls_label if cls_label.size > 0 else np.array([0, 0, 0, 1], dtype=np.float32)  # Default to no classes
        h5f.attrs['mpp'] = mpp
        h5f.attrs['patch_size'] = 224
        h5f.attrs['patch_level'] = mpp
        h5f.close()
        print(f"Saved {embedding.shape} features, mask, and attributes to {fname}.h5")
    except Exception as e:
        print(f"Error saving H5 file {fname}.h5: {e}")
        raise
    return fname

def create_embeddings(embeddings_dir, enc_name='resnet50_trunc_1024', dataset='BCSS-WSSS', batch_size=1, overwrite=True, assets_dir='./ckpts/'):
    print(f"\n=== Extracting Features for '{dataset}' via '{enc_name}' ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    start_time = time.time()
    if enc_name == 'resnet50_trunc_1024':
        model = resnet50_trunc_1024(pretrained=True)
        eval_t = eval_transforms(pretrained=True)
    elif enc_name == 'clip_RN50':
        import open_clip
        model, _, _ = open_clip.create_model_and_transforms('RN50', pretrained='openai')
        model = model.visual  # Use only the visual backbone
        eval_t = eval_transforms_clip(pretrained=True)
    elif enc_name == 'clip_ViTB32':
        import open_clip
        model, _, _ = open_clip.create_model_and_transforms('ViT-B/32', pretrained='openai')
        model = model.visual
        eval_t = eval_transforms_clip(pretrained=True)
    elif enc_name == 'dino_vits8':
        from nn_encoder_arch.vision_transformer import vit_small
        ckpt_path = os.path.join(assets_dir, 'dino_vits8.pth')
        assert os.path.isfile(ckpt_path)
        model = vit_small(patch_size=16)
        state_dict = torch.load(ckpt_path, map_location="cpu")['teacher']
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        eval_t = eval_transforms(pretrained=False)
    else:
        raise NotImplementedError(f"Model {enc_name} not supported")

    model = model.to(device)
    model.eval()
    print(f"Model loading and initialization took {time.time() - start_time:.2f} seconds")

    # Use the first 100 samples from BCSS-WSSS dataset
    slide_ids = []
    all_datasets = [
        BCSS_WSSSTrainingDataset(transform=None),
        BCSS_WSSS_TestDataset(split="test", transform=None),
        BCSS_WSSS_TestDataset(split="val", transform=None),
        BCSSWSSSDataset(mask_name="val/mask", transform=None),
        BCSSWSSSDataset(mask_name="test/mask", transform=None)
    ]
    total_samples = 0
    for dataset in all_datasets:
        for i in range(len(dataset)):
            if total_samples >= 100:
                break
            img_path, _, _, _ = dataset[i]
            slide_id = os.path.splitext(os.path.basename(img_path))[0]
            slide_ids.append(slide_id)
            total_samples += 1
        if total_samples >= 100:
            break

    dataset = PatchesDataset(slide_ids=slide_ids, transform=eval_t)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    os.makedirs(embeddings_dir, exist_ok=True)
    saved_files = []
    
    # Process 100 slides with progress bar
    print("\n" + "="*80)
    print("PROGRESS FEATURE EXTRACTION")
    print("="*80)
    with tqdm(total=len(slide_ids), desc="Extracting features", unit="slide") as pbar:
        for batch, mask, coord, slide_id, cls_label, mpp in dataloader:
            batch = batch.to(device)
            mask = mask.to(device) if mask is not None else None
            start_time = time.time()
            current_slide_id = slide_id[0] if isinstance(slide_id, (list, tuple)) else slide_id
            current_cls_label = cls_label[0] if cls_label.dim() > 1 else cls_label
            current_mpp = mpp[0] if isinstance(mpp, (list, tuple)) or mpp.dim() > 0 else mpp
            current_batch = batch
            current_mask = mask
            current_coord = coord
            fname = os.path.join(embeddings_dir, current_slide_id)
            saved_file = save_embeddings(model, fname, current_batch, current_mask, current_coord, current_cls_label, current_mpp, enc_name, overwrite=overwrite)
            if saved_file:
                saved_files.append(saved_file)
            print(f"Processing {current_slide_id} took {time.time() - start_time:.2f} seconds")
            pbar.update(1)
    
    print(f"\nCompleted extraction: {len(saved_files)} H5 files saved in {embeddings_dir}")
    if saved_files:
        print("Saved H5 files:", ", ".join([os.path.basename(f) for f in saved_files]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Configurations for feature extraction')
    parser.add_argument('--embeddings_dir', type=str, required=True, help='Base directory to save extracted features')
    parser.add_argument('--model_name', type=str, default='resnet50_trunc_1024', help='Model name (e.g., resnet50_trunc_1024, clip_RN50, clip_ViTB32, dino_vits8)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for feature extraction (default 1 for single patch)')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing .h5 files')
    args = parser.parse_args()

    create_embeddings(
        embeddings_dir=args.embeddings_dir,
        enc_name=args.model_name,
        dataset='BCSS-WSSS',
        batch_size=args.batch_size,
        overwrite=args.overwrite,
        assets_dir='./ckpts/'
    )
    
