import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2 as cv
from PIL import Image
import re
import traceback

# Base dir for BCSS-WSSS dataset (uncompressed directory)
BASE_DIR = r"D:\NghienCuu\NghienCuuPaper\Source_Code\data\data_BCSS-WSSS\BCSS-WSSS"

def check_dependencies():
    try:
        import cv2
        import torch
        import PIL
        print("Dependencies verified: cv2, torch, PIL are available")
    except ImportError as e:
        print(f"Missing dependency: {str(e)}")
        sys.exit(1)

class BCSS_WSSSTrainingDataset(Dataset):
    CLASSES = ["TUM", "STR", "LYM", "NEC"]
    NUM_CLASSES = 4  # Only TUM, STR, LYM, NEC; BACK is not included in labels

    def __init__(self, transform=None):
        super(BCSS_WSSSTrainingDataset, self).__init__()
        print(f"Initializing BCSSTrainingDataset with BASE_DIR: {BASE_DIR}")
        self.transform = transform
        try:
            self.get_images_and_labels()
            if self.cls_labels:
                min_val = float(min(self.cls_labels[0]))
                max_val = float(max(self.cls_labels[0]))
            else:
                min_val, max_val = 0.0, 0.0
            print(f"Training Dataset Structure: {len(self.img_paths)} images found in {os.path.join(BASE_DIR, 'training')}")
            print(f"Training Dataset Info: Total images = {len(self.img_paths)}, Labels range = {min_val}:{max_val}")
        except Exception as e:
            print(f"Error initializing BCSSTrainingDataset: {str(e)}\n{traceback.format_exc()}")
            raise

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        cls_label = self.cls_labels[index]
        try:
            img = cv.imread(img_path)
            if img is None:
                raise ValueError(f"Failed to decode image: {img_path}")
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error loading {img_path}: {str(e)}\n{traceback.format_exc()}")
            raise
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        cls_label = torch.tensor(cls_label, dtype=torch.float32)
        gt_label = torch.zeros(img.shape[-2:], dtype=torch.long)
        return os.path.basename(img_path), img, cls_label, gt_label

    def get_images_and_labels(self):
        self.img_paths = []
        self.cls_labels = []
        train_dir = os.path.join(BASE_DIR, 'training')
        print(f"Checking training directory: {train_dir}")

        if not os.path.exists(train_dir):
            raise ValueError(f"Training directory not found: {train_dir}")
        print(f"Training directory exists, listing files...")

        for filename in os.listdir(train_dir):
            if filename.endswith('.png'):
                img_path = os.path.join(train_dir, filename)
                fname = os.path.splitext(filename)[0]
                match = re.search(r'\[(\d{4})\]', fname)
                if match:
                    label_str = match.group(1)
                    try:
                        cls_label = np.array([int(x) for x in label_str], dtype=np.float32)
                        if np.sum(cls_label) == 0:
                            print(f"Warning: Zero label [0, 0, 0, 0] found for {filename}, skipping")
                            continue  # Skip images with no valid labels
                        self.img_paths.append(img_path)
                        self.cls_labels.append(cls_label)
                    except ValueError as ve:
                        print(f"Error converting label {label_str} for {filename}: {str(ve)}, skipping")
                        continue  # Skip images with invalid labels
                else:
                    print(f"Warning: No label found in filename {filename}, skipping")
                    continue  # Skip images without labels

        if not self.img_paths:
            raise ValueError("No valid training images with labels found in BCSS-WSSS/training/")
        print(f"Completed loading {len(self.img_paths)} training images")

class BCSS_WSSS_TestDataset(Dataset):
    CLASSES = ["TUM", "STR", "LYM", "NEC"]
    NUM_CLASSES = 4  # Only TUM, STR, LYM, NEC; BACK is not included in labels

    def __init__(self, split="test", transform=None):
        assert split in ["test", "val"], "split must be one of [test, val]"
        super(BCSS_WSSS_TestDataset, self).__init__()
        self.split = split
        self.transform = transform
        try:
            self.get_images_and_labels()
            if self.cls_labels:
                min_val = float(min(self.cls_labels[0]))
                max_val = float(max(self.cls_labels[0]))
            else:
                min_val, max_val = 0.0, 0.0
            print(f"{self.split.capitalize()} Dataset Structure: {len(self.img_paths)} images found in {os.path.join(BASE_DIR, self.split, 'img')}")
            if self.mask_paths:
                print(f"Mask Shape: {cv.imread(self.mask_paths[0], cv.IMREAD_GRAYSCALE).shape if self.mask_paths else 'N/A'}")
            print(f"{self.split.capitalize()} Dataset Info: Total images = {len(self.img_paths)}, Labels range = {min_val}:{max_val}")
        except Exception as e:
            print(f"Error initializing {self.split} Dataset: {str(e)}\n{traceback.format_exc()}")
            raise

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        cls_label = self.cls_labels[index]
        mask_path = self.mask_paths[index] if index < len(self.mask_paths) else None

        try:
            img = cv.imread(img_path)
            if img is None:
                raise ValueError(f"Failed to decode image: {img_path}")
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

            if mask_path:
                mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
                if mask is None or mask.size == 0:
                    raise ValueError(f"Failed to decode mask: {mask_path}")
            else:
                mask = np.zeros((224, 224), dtype=np.uint8)
        except Exception as e:
            raise ValueError(f"Error reading image {img_path}: {str(e)}\n{traceback.format_exc()}")

        if self.transform is not None:
            transformed = self.transform(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]

        cls_label = torch.tensor(cls_label, dtype=torch.float32)
        return os.path.basename(img_path), img, cls_label, mask

    def get_images_and_labels(self):
        self.img_paths = []
        self.mask_paths = []
        self.cls_labels = []
        img_dir = os.path.join(BASE_DIR, self.split, 'img')
        mask_dir = os.path.join(BASE_DIR, self.split, 'mask') if os.path.exists(os.path.join(BASE_DIR, self.split, 'mask')) else None

        if not os.path.exists(img_dir):
            raise ValueError(f"Image directory not found: {img_dir}")
        print(f"Image directory exists, listing files...")

        # Iterate over images in img_dir
        for filename in os.listdir(img_dir):
            if filename.endswith('.png'):
                img_path = os.path.join(img_dir, filename)
                self.img_paths.append(img_path)
                # Default label
                cls_label = np.zeros(self.NUM_CLASSES, dtype=np.float32)
                self.cls_labels.append(cls_label)
                print(f"Added {self.split} {filename} with default label {cls_label}")

                # Check for corresponding mask
                if mask_dir and os.path.exists(mask_dir):
                    mask_path = os.path.join(mask_dir, filename)
                    if os.path.exists(mask_path):
                        self.mask_paths.append(mask_path)
                        try:
                            with open(mask_path, 'rb') as mask_file:
                                mask = np.array(Image.open(mask_file))
                                unique_labels = np.unique(mask)
                                if unique_labels.size > 0 and any(l < self.NUM_CLASSES for l in unique_labels):
                                    cls_label = np.zeros(self.NUM_CLASSES, dtype=np.float32)
                                    cls_label[unique_labels[unique_labels < self.NUM_CLASSES]] = 1.0
                                    if np.sum(cls_label) == 0:
                                        print(f"Warning: No valid classes in mask for {filename}, keeping default label")
                                    else:
                                        self.cls_labels[-1] = cls_label
                                        print(f"Updated {filename} with mask labels {cls_label}")
                        except Exception as e:
                            print(f"Error processing mask {mask_path} for {filename}: {str(e)}, using default label")
                    else:
                        self.mask_paths.append(None)
                        print(f"No mask found for {filename}, using default label")
                else:
                    self.mask_paths.append(None)

        if not self.img_paths:
            raise ValueError(f"No {self.split} images found in {img_dir}")
        print(f"{self.split.capitalize()} Dataset Info: Total images = {len(self.img_paths)}")

class BCSSWSSSDataset(Dataset):
    CLASSES = ["TUM", "STR", "LYM", "NEC"]
    NUM_CLASSES = 4  # Only TUM, STR, LYM, NEC; BACK is not included in labels

    def __init__(self, mask_name="val/mask", transform=None, base_dir=BASE_DIR, split="val", pseudo_mask_dir=None, img_dir=None):
        super(BCSSWSSSDataset, self).__init__()
        print(f"Initializing BCSSWSSSDataset with mask_name: {mask_name}, BASE_DIR: {base_dir}, split: {split}")
        self.transform = transform
        self.mask_name = mask_name
        self.base_dir = base_dir
        self.split = split.lower()  # Ensure split is lowercase
        self.pseudo_mask_dir = pseudo_mask_dir  # Optional pseudo mask directory
        self.img_dir = img_dir if img_dir else os.path.join(base_dir, split, 'img')  # Use provided img_dir or default
        try:
            self.get_images_and_labels()
            if self.cls_labels:
                min_val = float(min(self.cls_labels[0]))
                max_val = float(max(self.cls_labels[0]))
            else:
                min_val, max_val = 0.0, 0.0
            print(f"WSSS Dataset Info: Total pairs = {len(self.img_paths)}, Labels range = {min_val}:{max_val}")
        except Exception as e:
            print(f"Error initializing BCSSWSSSDataset: {str(e)}\n{traceback.format_exc()}")
            raise

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        mask_path = self.mask_paths[index] 
        cls_label = self.cls_labels[index]

        try:
            img = cv.imread(img_path)
            if img is None:
                raise ValueError(f"Failed to decode image: {img_path}")
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE) if mask_path else None
            if mask is None and mask_path:
                raise ValueError(f"Failed to decode mask: {mask_path}")
        except Exception as e:
            raise ValueError(f"Error reading {img_path} or {mask_path}: {str(e)}\n{traceback.format_exc()}")

        if self.transform is not None:
            transformed = self.transform(image=img, mask=mask) if mask is not None else self.transform(image=img)
            img = transformed["image"]
            mask = transformed["mask"] if mask is not None else None

        cls_label = torch.tensor(cls_label, dtype=torch.float32)
        return os.path.basename(img_path), img, cls_label, mask

    def get_images_and_labels(self):
        self.img_paths = []
        self.mask_paths = []
        self.cls_labels = []

        # Check if image directory exists
        if not os.path.exists(self.img_dir):
            raise ValueError(f"Image directory not found: {self.img_dir}")

        # Define mask directory based on mask_name or split
        mask_dir = os.path.join(self.base_dir, self.mask_name) if os.path.exists(os.path.join(self.base_dir, self.mask_name)) else None
        if not mask_dir and self.split in ['val', 'test']:
            mask_dir = os.path.join(self.base_dir, self.split, 'mask')
        elif self.split == 'training' and self.pseudo_mask_dir and os.path.exists(self.pseudo_mask_dir):
            mask_dir = self.pseudo_mask_dir

        # List mask files if mask directory exists
        mask_files = []
        if mask_dir and os.path.exists(mask_dir):
            mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]

        # Collect images and corresponding masks
        for filename in os.listdir(self.img_dir):
            if filename.endswith('.png'):
                img_path = os.path.join(self.img_dir, filename)
                self.img_paths.append(img_path)
                cls_label = np.zeros(self.NUM_CLASSES, dtype=np.float32)
                self.cls_labels.append(cls_label)
                print(f"Added {filename} with default label {cls_label}")

                mask_path = None
                if mask_dir and filename in mask_files:
                    mask_path = os.path.join(mask_dir, filename)
                elif self.pseudo_mask_dir and os.path.exists(os.path.join(self.pseudo_mask_dir, filename)):
                    mask_path = os.path.join(self.pseudo_mask_dir, filename)
                
                self.mask_paths.append(mask_path)
                if mask_path:
                    try:
                        with open(mask_path, 'rb') as mask_file:
                            mask = np.array(Image.open(mask_file))
                            unique_labels = np.unique(mask)
                            if unique_labels.size > 0 and any(l < self.NUM_CLASSES for l in unique_labels):
                                cls_label = np.zeros(self.NUM_CLASSES, dtype=np.float32)
                                cls_label[unique_labels[unique_labels < self.NUM_CLASSES]] = 1.0
                                if np.sum(cls_label) == 0:
                                    print(f"Warning: No valid classes in mask for {filename}, keeping default label")
                                else:
                                    self.cls_labels[-1] = cls_label
                                    print(f"Updated {filename} with mask labels {cls_label}")
                    except Exception as e:
                        print(f"Error processing mask {mask_path} for {filename}: {str(e)}, using default label")
                else:
                    print(f"No mask found for {filename}, using default label")

        if not self.img_paths:
            raise ValueError(f"No samples found in {self.img_dir}")
        print(f"WSSS Dataset Info: Total pairs = {len(self.img_paths)}")

if __name__ == "__main__":
    check_dependencies()
    print(f"Starting dataset validation at {os.path.abspath(__file__)}")
    try:
        # Instantiate datasets to trigger initialization prints
        train_dataset = BCSS_WSSSTrainingDataset(transform=None)
        test_dataset = BCSS_WSSS_TestDataset(split="test", transform=None)
        val_dataset = BCSS_WSSS_TestDataset(split="val", transform=None)
        # Validate BCSSWSSSDataset with both val/mask and test/mask
        wsss_val_dataset = BCSSWSSSDataset(mask_name="val/mask", transform=None, split="val")
        wsss_test_dataset = BCSSWSSSDataset(mask_name="test/mask", transform=None, split="test")
        print("All datasets initialized successfully")
    except Exception as e:
        print(f"Failed to initialize datasets: {str(e)}\n{traceback.format_exc()}")
        sys.exit(1)
        