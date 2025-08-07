import os
import sys
import torch
import yaml
import numpy as np
from PIL import Image

from omegaconf import OmegaConf

# Add taming-transformers to path
sys.path.append("src/externals/taming-transformers")

# Disable gradient computation to save memory
torch.set_grad_enabled(False)

from taming.models.vqgan import VQModel, GumbelVQ

# Define or import these functions depending on your setup
# Placeholder implementations if not defined elsewhere
def download_image(url):
    from torchvision import transforms
    from PIL import Image
    import requests
    from io import BytesIO

    response = requests.get(url)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)

def stack_reconstructions(*images, titles=None):
    return images[0]  # placeholder — replace with actual image stacking function

titles = ["Input", "VQGAN"]  # Adjust based on what you want to show

def load_config(config_path, display=False):
    config = OmegaConf.load(config_path)
    if display:
        print(yaml.dump(OmegaConf.to_container(config)))
    return config

def load_vqgan(config, ckpt_path=None, is_gumbel=False):
    if is_gumbel:
        model = GumbelVQ(**config.model.params)
    else:
        model = VQModel(**config.model.params)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        model.load_state_dict(sd, strict=False)
    return model.eval()

def preprocess_vqgan(x):
    x = 2. * x - 1.
    return x

def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x

def reconstruct_with_vqgan(x, model):
	z_q, _, extras = model.encode(x)

	# The actual codebook indices are in extras[2]
	indices = extras[2]
	z = indices  # we'll treat this as the index map
	print(f"VQGAN --- {model.__class__.__name__}: latent shape: {z.shape}")
	xrec = model.decode(z_q)
	return xrec, z, z_q, indices
 
 
def reconstruction_pipeline(image_path, size=320):
    from torchvision import transforms
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])
 
    x_vqgan = transform(image).unsqueeze(0)
    
    # x_vqgan = preprocess(image_path, target_image_size=size, map_dalle=False)
    x_vqgan = x_vqgan.to(DEVICE)

    print(f"Input is of size: {x_vqgan.shape}")
    xrec, z, z_q, indices = reconstruct_with_vqgan(preprocess_vqgan(x_vqgan), model32x32)

    # Show input and reconstruction only
    img = stack_reconstructions(
        custom_to_pil(preprocess_vqgan(x_vqgan[0])),
        custom_to_pil(xrec[0]),
        titles=titles
    )
    return img, z, z_q, indices


# Fix for missing preprocess wrapper
def preprocess(image_tensor, target_image_size=320, map_dalle=False):
    from torchvision.transforms.functional import resize
    image_tensor = resize(image_tensor, [target_image_size, target_image_size])
    return image_tensor

if __name__ == "__main__":
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src/externals/taming-transformers')
    config_path = os.path.join(root_dir, "logs/vqgan_gumbel_f8/configs/model.yaml")
    check_point_path = os.path.join(root_dir, "logs/vqgan_gumbel_f8/checkpoints/last.ckpt")

    config32x32 = load_config(config_path, display=False)
    model32x32 = load_vqgan(config32x32, ckpt_path=check_point_path, is_gumbel=True).to(DEVICE)

    example = '/project/hnguyen2/mvu9/datasets/weakly_sup_segmentation_datasets/LUAD-HistoSeg/training/1224965-9156-45007-[1 0 0 1].png'
    output_img, z, z_q, indices = reconstruction_pipeline(example)
    output_img.save("reconstruction_output.png")

    # Optional debug:
    print(f"z (encoder output): {z.shape}")
    print(f"z_q (quantized): {z_q.shape}")
   
    print("Codebook indices (2D map):")
    print(indices[0].cpu().numpy().astype(int))  # shape: [H, W] 