from __future__ import print_function, division
import os
import torch
import numpy as np
from PIL import Image
import h5py
import cv2 as cv
from torch.utils.data import Dataset
from torchvision import transforms
import sys

# Add root directory to Python path
ROOT_FOLDER = r"D:\NghienCuu\NghienCuuPaper\Source_Code\WSSS-01\src\modified_externals\ViLa-PIP"
sys.path.insert(0, ROOT_FOLDER)

# Base dir for BCSS-WSSS dataset (uncompressed directory)
BASE_DIR = r"D:\NghienCuu\NghienCuuPaper\Source_Code\data\data_BCSS-WSSS\BCSS-WSSS"
EMBEDDINGS_DIR = r"D:\NghienCuu\NghienCuuPaper\Source_Code\WSSS-01\src\modified_externals\ViLa-PIP\features_extraction\test_feature_extraction"

def eval_transforms(pretrained=False):
    if pretrained:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)

    transforms_val = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]
    )

    return transforms_val

class Whole_Slide_Bag(Dataset):
    def __init__(self,
                 embeddings_dir=EMBEDDINGS_DIR,
                 pretrained=False,
                 custom_transforms=None,
                 target_patch_size=-1,
                 ):
        """
        Args:
            embeddings_dir (string): Path to directory containing .h5 files with features and coords.
            pretrained (bool): Use ImageNet transforms.
            custom_transforms (callable, optional): Optional transform to be applied on a sample.
            target_patch_size (int): Custom defined image size before embedding (not used for features).
        """
        self.pretrained = pretrained
        self.embeddings_dir = embeddings_dir
        if target_patch_size > 0:
            self.target_patch_size = (target_patch_size, target_patch_size)
        else:
            self.target_patch_size = None

        if not custom_transforms:
            self.roi_transforms = eval_transforms(pretrained=pretrained)
        else:
            self.roi_transforms = custom_transforms

        # Load H5 files and print their information
        self.h5_files = [f for f in os.listdir(embeddings_dir) if f.endswith('.h5')]
        self.h5_data = []
        print("\n=== H5 Files Information ===")
        for h5_file in self.h5_files:  # Print all H5 files
            h5_path = os.path.join(embeddings_dir, h5_file)
            slide_id = os.path.splitext(h5_file)[0]
            try:
                with h5py.File(h5_path, 'r') as f:
                    features = f['features'][:]
                    coords = f['coords'][:]
                    # Check for optional label in H5 attributes, expect 4 classes
                    label = f.attrs.get('label', np.zeros(4, dtype=np.float32))  # Default to 4 classes
                    if not (len(label) == 4 and label.dtype == np.float32):
                        print(f"Warning: Invalid label shape or type in H5 {h5_path}, defaulting to [0. 0. 0. 0.]")
                        label = np.zeros(4, dtype=np.float32)
                    self.h5_data.append({
                        'slide_id': slide_id,
                        'features': features,
                        'coords': coords,
                        'h5_path': h5_path,
                        'label': label
                    })
                    print(f"  Slide ID: {slide_id}")
                    print(f"    H5 Path: {h5_path}")
                    print(f"    Features Shape: {features.shape}")
                    print(f"    Coords Shape: {coords.shape}")
                    print(f"    Label: {label}")
            except Exception as e:
                print(f"Error loading H5 file {h5_path}: {e}")
        self.length = len(self.h5_data)
        if self.length == 0:
            print(f"Warning: No valid H5 files found in {embeddings_dir}")
            raise ValueError(f"No valid H5 files found in {embeddings_dir}")

        self.summary()

    def __len__(self):
        return self.length

    def summary(self):
        print("\n=== Whole_Slide_Bag Dataset Summary ===")
        print(f"  Total H5 files loaded: {self.length}")
        print("======================================")

    def __getitem__(self, idx):
        data = self.h5_data[idx]
        slide_id = data['slide_id']
        features = torch.from_numpy(data['features']).float()
        coord = torch.from_numpy(data['coords']).float()
        label = torch.from_numpy(data['label']).float()

        return features, coord, label, slide_id

class Whole_Slide_Bag_FP(Dataset):
    def __init__(self,
                 embeddings_dir=EMBEDDINGS_DIR,
                 pretrained=False,
                 custom_transforms=None,
                 custom_downsample=1,
                 target_patch_size=-1
                 ):
        """
        Args:
            embeddings_dir (string): Path to directory containing .h5 files with features and coords.
            pretrained (bool): Use ImageNet transforms.
            custom_transforms (callable, optional): Optional transform to be applied on a sample.
            custom_downsample (int): Custom defined downscale factor (overruled by target_patch_size).
            target_patch_size (int): Custom defined image size before embedding (not used for features).
        """
        self.pretrained = pretrained
        self.embeddings_dir = embeddings_dir
        if not custom_transforms:
            self.roi_transforms = eval_transforms(pretrained=pretrained)
        else:
            self.roi_transforms = custom_transforms
        if target_patch_size > 0:
            self.target_patch_size = (target_patch_size, target_patch_size)
        elif custom_downsample > 1:
            self.target_patch_size = (224 // custom_downsample, 224 // custom_downsample)
        else:
            self.target_patch_size = None

        # Load H5 files and print their information
        self.h5_files = [f for f in os.listdir(embeddings_dir) if f.endswith('.h5')]
        self.h5_data = []
        print("\n=== H5 Files Information ===")
        for h5_file in self.h5_files:  # Print all H5 files
            h5_path = os.path.join(embeddings_dir, h5_file)
            slide_id = os.path.splitext(h5_file)[0]
            try:
                with h5py.File(h5_path, 'r') as f:
                    features = f['features'][:]
                    coords = f['coords'][:]
                    patch_level = f['coords'].attrs.get('patch_level', 0)
                    patch_size = f['coords'].attrs.get('patch_size', 224)
                    # Check for optional label in H5 attributes, expect 4 classes
                    label = f.attrs.get('label', np.zeros(4, dtype=np.float32))  # Default to 4 classes
                    if not (len(label) == 4 and label.dtype == np.float32):
                        print(f"Warning: Invalid label shape or type in H5 {h5_path}, defaulting to [0. 0. 0. 0.]")
                        label = np.zeros(4, dtype=np.float32)
                    self.h5_data.append({
                        'slide_id': slide_id,
                        'features': features,
                        'coords': coords,
                        'h5_path': h5_path,
                        'patch_level': patch_level,
                        'patch_size': patch_size,
                        'label': label
                    })
                    print(f"  Slide ID: {slide_id}")
                    print(f"    H5 Path: {h5_path}")
                    print(f"    Features Shape: {features.shape}")
                    print(f"    Coords Shape: {coords.shape}")
                    print(f"    Patch Level: {patch_level}")
                    print(f"    Patch Size: {patch_size}")
                    print(f"    Label: {label}")
            except Exception as e:
                print(f"Error loading H5 file {h5_path}: {e}")
        self.length = len(self.h5_data)
        if self.length == 0:
            print(f"Warning: No valid H5 files found in {embeddings_dir}")
            raise ValueError(f"No valid H5 files found in {embeddings_dir}")

        self.summary()

    def __len__(self):
        return self.length

    def summary(self):
        print("\n=== Whole_Slide_Bag_FP Dataset Summary ===")
        print(f"  Total H5 files loaded: {self.length}")
        print("======================================")

    def __getitem__(self, idx):
        data = self.h5_data[idx]
        slide_id = data['slide_id']
        features = torch.from_numpy(data['features']).float()
        coord = torch.from_numpy(data['coords']).float()
        label = torch.from_numpy(data['label']).float()

        # Return dummy image for compatibility (not used for features)
        img = torch.zeros(1, 3, 224, 224)

        return img, features, coord, label, slide_id

class Dataset_All_Bags(Dataset):
    def __init__(self, embeddings_dir=EMBEDDINGS_DIR):
        # Load slide IDs from H5 files
        self.slide_ids = [os.path.splitext(f)[0] for f in os.listdir(embeddings_dir) if f.endswith('.h5')]

    def __len__(self):
        return len(self.slide_ids)

    def __getitem__(self, idx):
        return self.slide_ids[idx]

if __name__ == "__main__":
    print("Starting dataset_h5.py validation")
    try:
        # Instantiate Whole_Slide_Bag
        print("\n=== Instantiating Whole_Slide_Bag ===")
        whole_slide_bag = Whole_Slide_Bag(embeddings_dir=EMBEDDINGS_DIR, pretrained=False)

        # Instantiate Whole_Slide_Bag_FP
        print("\n=== Instantiating Whole_Slide_Bag_FP ===")
        whole_slide_bag_fp = Whole_Slide_Bag_FP(embeddings_dir=EMBEDDINGS_DIR, pretrained=False)

        # Instantiate Dataset_All_Bags
        print("\n=== Instantiating Dataset_All_Bags ===")
        dataset_all_bags = Dataset_All_Bags(embeddings_dir=EMBEDDINGS_DIR)
        print("\nDataset_All_Bags Summary:")
        print(f"  Total slide IDs: {len(dataset_all_bags)}")
        print("  Slide IDs (up to 5):")
        for i, slide_id in enumerate(dataset_all_bags):
            print(f"    {slide_id}")
            if i >= 4:  # Limit to 5 examples
                break

        print("\nAll datasets initialized successfully")
    except Exception as e:
        print(f"Failed to initialize datasets: {e}")
        import traceback
        traceback.print_exc()
        