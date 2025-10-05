import torch
import os
import argparse
from patch_extraction_utils import create_embeddings
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(description='Configurations for feature extraction')
parser.add_argument('--embeddings_dir', type=str, required=True, help='Base directory to save extracted features')
parser.add_argument('--model_name', type=str, default='resnet50_trunc', help='Model name (e.g., clip_RN50, resnet50_trunc, resnet34_trunc_768, clip_ViTB32, dino_vits8)')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for feature extraction (default 1 for single patch)')
parser.add_argument('--overwrite', action='store_true', help='Overwrite existing .h5 files')
args = parser.parse_args()

embeddings_dir = args.embeddings_dir
model_name = args.model_name
batch_size = args.batch_size
overwrite = args.overwrite

os.makedirs(embeddings_dir, exist_ok=True)

create_embeddings(
    embeddings_dir=embeddings_dir,
    enc_name=model_name,
    dataset='BCSS-WSSS',
    batch_size=batch_size,
    overwrite=overwrite,
    assets_dir='./ckpts/'
)
