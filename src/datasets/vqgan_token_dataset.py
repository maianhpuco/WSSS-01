import os
import re
import torch
import sys
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from omegaconf import OmegaConf

# Add taming-transformers to path
sys.path.append("src/externals/taming-transformers")
from taming.models.vqgan import GumbelVQ

class VQGANWeakSegDataset(Dataset):
    def __init__(self, image_dir, config_path, ckpt_path, image_size=320, device='cpu', return_mask=False, mask_dir=None):
        super().__init__()
        self.device = device
        self.return_mask = return_mask
        self.mask_dir = mask_dir

        self.image_paths = sorted([
            os.path.join(image_dir, f) for f in os.listdir(image_dir)
            if f.endswith(".png") and ("-[" in f or return_mask)
        ])

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

        # Load VQGAN
        config = OmegaConf.load(config_path)
        self.model = GumbelVQ(**config.model.params).to(device)
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)["state_dict"]
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

    def __len__(self):
        return len(self.image_paths)

    def parse_label_from_filename(self, filename):
        match = re.search(r'\[(.*?)\]', filename)
        if match:
            label_str = match.group(1)
            return torch.tensor([int(x) for x in label_str.strip().split()], dtype=torch.float32)
        else:
            return torch.zeros(4)  # fallback for unlabeled files

    def preprocess_vqgan(self, x):
        return 2. * x - 1.

    def load_mask(self, filename):
        mask_filename = filename.split("-[")[0] + ".png" if "-[" in filename else filename
        mask_path = os.path.join(self.mask_dir, mask_filename)
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        mask = Image.open(mask_path).convert("L")
        return self.transform(mask).squeeze(0).long()

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        filename = os.path.basename(image_path)

        # Load and preprocess image
        img = Image.open(image_path).convert("RGB")
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        label = self.parse_label_from_filename(filename)

        with torch.no_grad():
            z = self.preprocess_vqgan(img_tensor)
            _, _, extras = self.model.encode(z)
            indices = extras[2]  # [1, H, W]

        if self.return_mask:
            mask = self.load_mask(filename)
            return indices.squeeze(0).long(), label, mask
        else:
            return indices.squeeze(0).long(), label
