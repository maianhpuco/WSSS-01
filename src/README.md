## 🛠️ Installation

### Requirements
- Python 3.8+
- PyTorch 1.9+
- CUDA 11.0+ (for GPU training)

### Environment Setup

#### Using requirements.txt
```bash
# Create virtual environment
conda create -n pbip python=3.11.4
conda activate pbip

# Install exact dependencies (recommended for reproducibility)
pip install -r requirements.txt
```

## 📊 Dataset

This project uses the **BCSS (Breast Cancer Semantic Segmentation)** dataset with 5 tissue classes:

| Class | Description | Color |
|-------|-------------|-------|
| TUM | Tumor | 🔴 Red |
| STR | Stroma | 🟢 Green |
| LYM | Lymphocyte | 🔵 Blue |
| NEC | Necrosis | 🟣 Purple |
| BACK | Background | ⚪ White |

### Data Structure
```
data/
├── BCSS-WSSS/
│   ├── train/
│   │   └── *.png  # Training images with class labels in filename
│   ├── test/
│   │   ├── img/   # Test images
│   │   └── mask/  # Ground truth masks
│   └── valid/
│       ├── img/   # Validation images
│       └── mask/  # Ground truth masks
```

# Weakly Supervised Semantic Segmentation (WSSS) with ViLa-PIP

This repository contains the source code and scripts for a Weakly Supervised Semantic Segmentation (WSSS) project using the ViLa-PIP framework. The project focuses on feature extraction, model training, and inference for binary masks and CAM heatmaps on the BCSS dataset. Last updated: 01:54 AM +07, Monday, October 06, 2025.

## Overview
- **Feature Extraction**: Extracts features from the training dataset using a pre-trained model.
- **Training**: Trains a classification model with a specified configuration.
- **Inference**: Generates binary masks and CAM heatmaps for segmentation tasks.

## Prerequisites
- **Python**: Version 3.7 or higher
- **Dependencies**:
  - `torch` (with CUDA support recommended)
  - `numpy`
  - `omegaconf`
  - `h5py`
  - `pillow`
  - `tqdm`
  - `albumentations`
  - `matplotlib`
- **Hardware**: CUDA-enabled GPU (recommended for training and inference)
- **Environment**: Virtual environment (e.g., using `venv` or `conda`)

### Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>

2. Install dependencies:
pip install torch torchvision numpy omegaconf h5py pillow tqdm albumentations matplotlib



### Source Code Structure
|_ src
  |_ modified_externals
    |_ ViLa-PIP
      |_ features_extraction/
      | |_ patch_extraction.py
      |_ train/
      | |_ train_training_dataset.py
      |_ inference/
      | |_ inference_binary_mask.py
      | |_ inference_cam.py
      |_ work_dirs/
      | |_ bcss_wsss/
      |   |_ classification/
      |     |_ config.yaml
      |_ datasets/
      | |_ BCSS-WSSS/
      |   |_ training/
      |   | |_ img/
      |   |   |_ TCGA-A1-A0SK-DX1_xmin45749_ymin25055_MPP-0.2500+0[1101]
      |   |_ val/
      |   | |_ img/
      |   | | |_ TCGA-D8-A27F-DX1_xmin98787_ymin6725_MPP-0.2500+0
      |   | |_ mask/
      |   |   |_ TCGA-D8-A27F-DX1_xmin98787_ymin6725_MPP-0.2500+0
      |   |_ test/
      |     |_ img/
      |     | |_ TCGA-EW-A1PB-DX1_xmin57214_ymin25940_MPP-0.2500+0
      |     |_ mask/
      |       |_ TCGA-EW-A1PB-DX1_xmin57214_ymin25940_MPP-0.2500+0
      |_ model/
      | |_ model_ViLa_MIL.py
      | |_ projector.py
      | |_ model.py
      |_ utils/
        |_ fgbg_feature.py
        |_ contrast_loss.py
        |_ hierarchical_utils.py


### Running the Commands
1. Feature Extraction for Folder Training:
Extracts features from the training dataset using a pre-trained resnet50_trunc_1024 model.
python path\features_extraction\patch_extraction.py --embeddings_dir path\features_extract\and\VA\lion\dataset_features_extraction --model_name resnet50_trunc_1024 --batch_size 1 --overwrite


2. Train:
Trains the model using the provided configuration file on GPU 0.
python path\train\train_training_dataset.py --config path\work_dirs\bcss_wsss\classification\config.yaml --gpu 0

3. Inference Binary Mask:
python path\inference\inference_binary_mask.py

4. Inference CAM Heatmap:
python path\inference\inference_cam.py