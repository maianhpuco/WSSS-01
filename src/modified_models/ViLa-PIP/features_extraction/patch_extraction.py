import torch
import os
import argparse
from patch_extraction_utils import create_embeddings
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(description='Configurations for feature extraction')
parser.add_argument('--embeddings_dir', type=str, required=True, help='Base directory to save extracted features')
parser.add_argument('--model_name', type=str, default='resnet50_trunc_1024', help='Model name (e.g., resnet50_trunc_1024, clip_RN50, clip_ViTB32, dino_vits8)')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for feature extraction (default 1 for single patch)')
parser.add_argument('--overwrite', action='store_true', help='Overwrite existing .h5 files')
args = parser.parse_args()

embeddings_dir = args.embeddings_dir
model_name = args.model_name
batch_size = args.batch_size
overwrite = args.overwrite

# Ensure the embeddings directory exists
os.makedirs(embeddings_dir, exist_ok=True)

# Call the create_embeddings function with the updated model name
create_embeddings(
    embeddings_dir=embeddings_dir,
    enc_name=model_name,
    dataset='BCSS-WSSS',
    batch_size=batch_size,
    overwrite=overwrite,
    assets_dir='./ckpts/'
)
