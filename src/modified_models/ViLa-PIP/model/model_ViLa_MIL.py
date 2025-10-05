from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from open_clip.tokenizer import SimpleTokenizer as _Tokenizer
import warnings
import math
import h5py
import numpy as np
import os
import sys

ROOT_FOLDER = r"D:\NghienCuu\NghienCuuPaper\Source_Code\WSSS-01\src\modified_externals\ViLa-PIP"
sys.path.insert(0, ROOT_FOLDER)
from model.projector import PLIPProjector
from model.model_utils import Attn_Net_Gated

logger = logging.getLogger(__name__)
_tokenizer = _Tokenizer()

# Define H5 directories
H5_TRAIN_DIR = r"D:\NghienCuu\NghienCuuPaper\Source_Code\WSSS-01\src\modified_externals\ViLa-PIP\features_extraction\dataset_features_extraction"
H5_TEST_DIR = r"D:\NghienCuu\NghienCuuPaper\Source_Code\WSSS-01\src\modified_externals\ViLa-PIP\features_extraction\test_feature_extraction"
H5_VAL_DIR = r"D:\NghienCuu\NghienCuuPaper\Source_Code\WSSS-01\src\modified_externals\ViLa-PIP\features_extraction\val_feature_extraction"

BASE_DIR = r"D:\NghienCuu\NghienCuuPaper\Source_Code\data\data_BCSS-WSSS\BCSS-WSSS"

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = nn.Linear(512, 512).float()
        self.dtype = self.transformer.weight.dtype if hasattr(self.transformer, 'weight') else torch.float32

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)]
        x = self.text_projection(x)
        print(f"TextEncoder output shape: {x.shape}")
        return x.clone()

class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = 16
        ctx_init = ""
        dtype = clip_model.dtype if hasattr(clip_model, 'dtype') else torch.float32
        ctx_dim = clip_model.ln_final.weight.shape[0]

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = open_clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1:1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        self.ctx = nn.Parameter(ctx_vectors)
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [name for name in classnames]
        tokenized_prompts = torch.cat([open_clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1:-n_ctx, :])
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens
        self.class_token_position = "end"

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        prefix = self.token_prefix
        suffix = self.token_suffix
        if self.class_token_position == "end":
            prompts = torch.cat([prefix, ctx, suffix], dim=1)
        else:
            raise ValueError
        return prompts

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.
    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

class ViLa_MIL_Model(nn.Module):
    def __init__(self, config, num_classes=4):
        super(ViLa_MIL_Model, self).__init__()
        self.loss_ce = nn.CrossEntropyLoss()
        self.loss_bce = nn.BCEWithLogitsLoss()
        self.num_classes = num_classes
        self.L = 224 * 224  # Total number of pixels
        self.D = 512  # Feature dimension
        self.K = (224 // 16) * (224 // 16) + 1  # Number of patches + 1 (CLS token) = 197
        self.patch_grid = 14  # 224 / 16 = 14

        clip_model, _, _ = open_clip.create_model_and_transforms('RN50', pretrained='openai', quick_gelu=True)
        self.backbone = clip_model.visual
        self.backbone.to(dtype=torch.float32)
        self.backbone.norm = nn.LayerNorm(self.D)

        self.backbone_projection = nn.Linear(1024, self.D).float()
        self.attention_V = nn.Sequential(nn.Linear(self.D, 256), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(self.D, 256), nn.Sigmoid())
        self.attention_weights = nn.Linear(256, self.num_classes)
        self.attention_net = Attn_Net_Gated(L=self.D, D=256, dropout=False, n_classes=self.num_classes, num_patches=self.K)

        self.prompt_learner = PromptLearner(config.text_prompt, clip_model.float())
        self.text_encoder = TextEncoder(clip_model.float())
        local_model_path = r"D:\NghienCuu\NghienCuuPaper\Source_Code\WSSS-01\pretrained_models\vinid_plip"
        self.projector = PLIPProjector(local_model_path=local_model_path)

        self.norm = nn.LayerNorm(self.D)
        self.cross_attention_1 = nn.MultiheadAttention(embed_dim=self.D, num_heads=8, batch_first=True)
        self.cross_attention_2 = nn.MultiheadAttention(embed_dim=self.D, num_heads=8, batch_first=True)

        self.prototype_number = config.prototype_number
        self.learnable_image_center = nn.Parameter(torch.randn(self.prototype_number, 1, self.D))
        trunc_normal_(self.learnable_image_center, std=0.02)

        self.input_projection = nn.Linear(self.D, self.D, bias=False).float()
        self.fg_bg_classifier = nn.Linear(self.D, 1)
        self.text_projection = nn.Linear(512, self.D).float()
        self.coord_projection = nn.Linear(2, self.D).float()
        
        # Initialize l_fea as a learnable parameter
        self.l_fea = nn.Parameter(torch.randn(num_classes, self.D))
    
    def _extract_patched_features(self, x):
        device = next(self.parameters()).device
        x = x.to(device).float()  # [batch_size, 3, 224, 224]
        features = self.backbone(x)  # [batch_size, 197, 1024]
        features = self.backbone_projection(features)  # [batch_size, 197, 512]
        print(f"Extracted patched features shape: {features.shape}")
        return features

    def forward(self, x, labels=None, fg_bg_labels=None, h5_path=None, coords=None, img_name=None):
        device = next(self.parameters()).device
        batch_size = x.shape[0]
        print(f"Input x shape: {x.shape}")

        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"Warning: Input tensor contains NaN or inf values: {x}")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

        features = None
        if isinstance(h5_path, (list, tuple)):
            features_list = []
            all_h5_valid = True
            for path in h5_path:
                is_training = 'training' in path.lower() or 'dataset_features_extraction' in path.lower()
                is_test = 'test' in path.lower() or 'test_feature_extraction' in path.lower()
                is_val = 'val' in path.lower() or 'val_feature_extraction' in path.lower()
                dir_name = H5_TRAIN_DIR if is_training else H5_TEST_DIR if is_test else H5_VAL_DIR if is_val else H5_TRAIN_DIR
                base_name = os.path.splitext(os.path.basename(path))[0]
                suffix = ''.join(filter(lambda x: x.startswith('[') and x.endswith(']'), path.split(base_name)[-1].split('.')))
                h5_path_adjusted = os.path.join(dir_name, f"{base_name}{suffix}.h5" if suffix else f"{base_name}.h5")
                print(f"Adjusted H5 path from {path} to {h5_path_adjusted} (is_training: {is_training}, is_test: {is_test}, is_val: {is_val})")
                
                if not os.path.exists(h5_path_adjusted):
                    # Try alternative directories
                    other_dirs = [H5_TRAIN_DIR, H5_TEST_DIR, H5_VAL_DIR]
                    found = False
                    for other_dir in other_dirs:
                        alt_h5_path = os.path.join(other_dir, os.path.basename(h5_path_adjusted))
                        if os.path.exists(alt_h5_path):
                            h5_path_adjusted = alt_h5_path
                            print(f"Found H5 file in alternative directory: {h5_path_adjusted}")
                            found = True
                            break
                    if not found:
                        print(f"H5 file not found: {h5_path_adjusted}. Falling back to backbone features.")
                        all_h5_valid = False
                        break
                try:
                    with h5py.File(h5_path_adjusted, 'r') as h5:
                        feat = torch.from_numpy(h5['features'][:]).float().to(device)
                        print(f"H5 features shape: {feat.shape}, path: {h5_path_adjusted}")
                        h5_label = h5.attrs.get('label', None)
                        if h5_label is not None and labels is None and is_training:
                            labels = torch.tensor(h5_label[:self.num_classes], dtype=torch.float32).unsqueeze(0).to(device)
                            print(f"Loaded cls_label from H5: {labels.shape}, value: {labels}")
                        elif h5_label is None and labels is None and (is_test or is_val):
                            print(f"No label in H5 {h5_path_adjusted} (test/val), using default or provided labels")
                        
                        # Handle different feature shapes
                        if feat.dim() == 1 and feat.shape[0] == 1024:
                            feat = feat.unsqueeze(0).unsqueeze(0)  # [1, 1, 1024]
                            feat = self.backbone_projection(feat)  # [1, 1, 512]
                        elif feat.dim() == 2 and feat.shape[0] == 197 and feat.shape[1] == 1024:
                            feat = feat.unsqueeze(0)  # [1, 197, 1024]
                            feat = self.backbone_projection(feat)  # [1, 197, 512]
                        elif feat.dim() == 2 and feat.shape[1] == 197 and feat.shape[0] == 1024:
                            feat = feat.transpose(0, 1).unsqueeze(0)  # [1, 197, 1024]
                            feat = self.backbone_projection(feat)  # [1, 197, 512]
                        elif feat.dim() == 2 and feat.shape[1] == 512:
                            feat = feat.unsqueeze(1).expand(-1, self.K, -1)  # [batch_size, 197, 512]
                        else:
                            print(f"Unexpected H5 features shape {feat.shape}, expected [1024], [197, 1024], [1024, 197], or [batch_size, 512]. Falling back to backbone features.")
                            all_h5_valid = False
                            break
                        print(f"Projected H5 features shape: {feat.shape}")
                        features_list.append(feat)
                except Exception as e:
                    print(f"Error loading H5 file {h5_path_adjusted}: {str(e)}. Falling back to backbone features.")
                    all_h5_valid = False
                    break
            if all_h5_valid and len(features_list) == batch_size:
                features = torch.cat(features_list, dim=0)
                print(f"Concatenated H5 features shape: {features.shape}")
                if coords is not None and coords.shape[0] == batch_size and coords.shape[1] == 2:
                    coords_expanded = coords.unsqueeze(1).expand(-1, features.shape[1], -1)
                    coords_projected = self.coord_projection(coords_expanded)
                    features = features.clone() + coords_projected * 0.01
            else:
                print(f"Falling back to backbone features for batch due to invalid H5 files.")
                features = self._extract_patched_features(x)
        else:
            if h5_path and os.path.exists(h5_path):
                is_training = 'training' in h5_path.lower() or 'dataset_features_extraction' in h5_path.lower()
                is_test = 'test' in h5_path.lower() or 'test_feature_extraction' in h5_path.lower()
                is_val = 'val' in h5_path.lower() or 'val_feature_extraction' in h5_path.lower()
                dir_name = H5_TRAIN_DIR if is_training else H5_TEST_DIR if is_test else H5_VAL_DIR if is_val else H5_TRAIN_DIR
                base_name = os.path.splitext(os.path.basename(h5_path))[0]
                suffix = ''.join(filter(lambda x: x.startswith('[') and x.endswith(']'), h5_path.split(base_name)[-1].split('.')))
                h5_path_adjusted = os.path.join(dir_name, f"{base_name}{suffix}.h5" if suffix else f"{base_name}.h5")
                print(f"Adjusted H5 path from {h5_path} to {h5_path_adjusted}")
                try:
                    with h5py.File(h5_path_adjusted, 'r') as h5:
                        features = torch.from_numpy(h5['features'][:]).float().to(device)
                        h5_label = h5.attrs.get('label', None)
                        if h5_label is not None and labels is None and is_training:
                            labels = torch.tensor(h5_label[:self.num_classes], dtype=torch.float32).unsqueeze(0).to(device)
                            print(f"Loaded cls_label from H5: {labels.shape}, value: {labels}")
                        elif h5_label is None and labels is None and (is_test or is_val):
                            print(f"No label in H5 {h5_path_adjusted} (test/val), using default or provided labels")
                        print(f"H5 features shape: {features.shape}")
                        if features.dim() == 1 and features.shape[0] == 1024:
                            features = self.backbone_projection(features.unsqueeze(0))
                            features = features.unsqueeze(1).expand(batch_size, self.K, -1)
                        elif features.dim() == 2 and features.shape[0] == 197 and features.shape[1] == 1024:
                            features = self.backbone_projection(features)
                            features = features.unsqueeze(0).expand(batch_size, -1, -1)
                        elif features.dim() == 2 and features.shape[1] == 197 and features.shape[0] == 1024:
                            features = self.backbone_projection(features.transpose(0, 1))
                            features = features.unsqueeze(0).expand(batch_size, -1, -1)
                        elif features.dim() == 2 and features.shape[1] == 512:
                            features = features.unsqueeze(1).expand(batch_size, self.K, -1)
                        else:
                            raise ValueError(f"Unexpected H5 features shape {features.shape}, expected [1024], [197, 1024], [1024, 197], or [batch_size, 512]")
                        print(f"Projected H5 features shape: {features.shape}")
                        if coords is not None and coords.shape[0] == batch_size and coords.shape[1] == 2:
                            coords_expanded = coords.unsqueeze(1).expand(-1, features.shape[1], -1)
                            coords_projected = self.coord_projection(coords_expanded)
                            features = features.clone() + coords_projected * 0.01
                except Exception as e:
                    print(f"Error loading H5 file {h5_path_adjusted}: {str(e)}. Falling back to backbone features.")
                    features = self._extract_patched_features(x)
            else:
                print(f"H5 file not found or not provided: {h5_path}. Using backbone features.")
                features = self._extract_patched_features(x)

        # Handle backbone features
        print(f"Extracted patched features shape: {features.shape}")
        if torch.isnan(features).any() or torch.isinf(features).any():
            print(f"Warning: Backbone features contain NaN or inf values: {features}")
            features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

        # Adjust feature shapes to match expected dimensions
        if features.dim() == 2:
            if features.shape[1] == 512:
                features = features.unsqueeze(1).expand(-1, self.K, -1)  # [batch_size, 197, 512]
            elif features.shape[1] == 1024:
                features = self.backbone_projection(features)  # [batch_size, 512]
                features = features.unsqueeze(1).expand(-1, self.K, -1)  # [batch_size, 197, 512]
            else:
                raise ValueError(f"Unexpected global features shape {features.shape}, expected [batch_size, 512] or [batch_size, 1024]")
        elif features.dim() == 3:
            if features.shape[1] == self.K and features.shape[2] == 512:
                features = features
            elif features.shape[1] == self.K and features.shape[2] == 1024:
                features = self.backbone_projection(features)  # [batch_size, 197, 512]
            else:
                raise ValueError(f"Unexpected patch features shape {features.shape}, expected [batch_size, 197, 512] or [batch_size, 197, 1024]")
        else:
            raise ValueError(f"Unexpected features dimension {features.dim()}, expected 2 or 3")

        self.features = features
        print(f"Final features shape: {features.shape}")

        text_prompts = self.prompt_learner()
        print(f"text_prompts shape: {text_prompts.shape}")
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        text_features = self.text_encoder(text_prompts, tokenized_prompts)
        print(f"text_features before projection shape: {text_features.shape}")
        text_features = self.text_projection(text_features.clone().detach())
        print(f"text_features after projection shape: {text_features.shape}")

        compents_list = []
        try:
            for i in range(self.prototype_number):
                prototype = self.learnable_image_center[i:i + 1].expand(batch_size, -1, -1)
                print(f"prototype {i} shape: {prototype.shape}")
                compents, _ = self.cross_attention_1(prototype, features, features)
                print(f"compents {i} shape: {compents.shape}")
                compents = self.norm(compents + prototype)
                compents_list.append(compents)
            compents = torch.mean(torch.stack(compents_list, dim=0), dim=0)
            print(f"compents mean shape: {compents.shape}")
        except Exception as e:
            print(f"Warning: Failed to compute compents: {e}, using dummy values")
            compents = torch.zeros(batch_size, 1, self.D, device=device)

        H = compents
        print(f"H shape: {H.shape}")

        try:
            A, _ = self.attention_net(features)
            print(f"A shape: {A.shape}")
            if A.shape[1] != self.K:
                print(f"Warning: Attention weights num_patches ({A.shape[1]}) does not match expected K ({self.K}). This may indicate a misconfiguration in Attn_Net_Gated. Using computed num_patches.")
            A = F.softmax(A, dim=1)
        except Exception as e:
            print(f"Warning: Failed to compute attention weights: {e}, using dummy values")
            A = torch.zeros(batch_size, self.K, self.num_classes, device=device)

        if A.shape[1] <= features.shape[1]:
            image_features = torch.bmm(A.permute(0, 2, 1), features[:, :A.shape[1], :])
        else:
            print(f"Warning: A.shape[1] ({A.shape[1]}) exceeds features.shape[1] ({features.shape[1]}). Padding features.")
            padding = torch.zeros(batch_size, A.shape[1] - features.shape[1], self.D, device=device)
            features_padded = torch.cat((features, padding), dim=1)
            image_features = torch.bmm(A.permute(0, 2, 1), features_padded)
        print(f"image_features shape: {image_features.shape}")

        if A.shape[1] <= features.shape[1]:
            image_context = torch.cat((H, features[:, :A.shape[1], :]), dim=1)
        else:
            print(f"Warning: A.shape[1] ({A.shape[1]}) exceeds features.shape[1] ({features.shape[1]}). Padding features for image_context.")
            padding = torch.zeros(batch_size, A.shape[1] - features.shape[1], self.D, device=device)
            features_padded = torch.cat((features, padding), dim=1)
            image_context = torch.cat((H, features_padded), dim=1)
        print(f"image_context shape: {image_context.shape}")
        text_context_input = text_features.unsqueeze(1)
        print(f"text_context_input shape: {text_context_input.shape}")

        image_context_expanded = image_context.repeat_interleave(self.num_classes, dim=0)
        print(f"image_context_expanded shape: {image_context_expanded.shape}")
        text_context_input = text_context_input.unsqueeze(0).expand(batch_size, -1, -1, -1)
        print(f"Expanded text_context_input shape: {text_context_input.shape}")
        try:
            text_context_features, _ = self.cross_attention_2(
                text_context_input.reshape(batch_size * self.num_classes, 1, self.D),
                image_context_expanded,
                image_context_expanded)
            print(f"text_context_features shape: {text_context_features.shape}")
            text_context_features = text_context_features.reshape(batch_size, self.num_classes, 1, self.D)
            text_features = text_context_features.squeeze(2).mean(dim=0) + text_features.clone().detach()
            print(f"text_features after context shape: {text_features.shape}")
        except Exception as e:
            print(f"Warning: Failed to compute text_context_features: {e}, using dummy values")
            text_context_features = torch.zeros(batch_size * self.num_classes, 1, self.D, device=device)
            text_context_features = text_context_features.reshape(batch_size, self.num_classes, 1, self.D)
            text_features = text_features.clone().detach()

        text_features = self.projector.TextMLP(text_features)
        print(f"text_features after TextMLP shape: {text_features.shape}")
        logits = self.projector.forward(image_features, text_features)
        print(f"logits shape: {logits.shape}")

        fg_bg_logits = self.fg_bg_classifier(H.squeeze(1))
        print(f"fg_bg_logits shape: {fg_bg_logits.shape}")
        fg_bg_prob = torch.sigmoid(fg_bg_logits)

        loss = None
        try:
            if labels is not None and fg_bg_labels is not None:
                labels_indices = torch.argmax(labels, dim=1) if labels.dim() == 2 else labels
                ce_loss = self.loss_ce(logits, labels_indices)
                bce_loss = self.loss_bce(fg_bg_logits, fg_bg_labels.float())
                loss = ce_loss + bce_loss
            else:
                labels_indices = torch.zeros(batch_size, dtype=torch.long, device=device)
                ce_loss = self.loss_ce(logits, labels_indices)
                bce_loss = self.loss_bce(fg_bg_logits, torch.zeros(batch_size, 1, device=device))
                loss = ce_loss + bce_loss
        except Exception as e:
            print(f"Warning: Failed to compute loss: {e}, using None")
            loss = None

        Y_prob = F.softmax(logits, dim=1)
        print(f"Y_prob shape: {Y_prob.shape}")
        Y_hat = torch.argmax(Y_prob, dim=1)
        print(f"Y_hat shape: {Y_hat.shape}")

        # Generate CAMs for four classifiers
        cam1 = self._generate_cam(features, A)
        cam2 = cam1
        cam3 = cam1
        cam4 = cam1

        return Y_prob, cam1, Y_prob, cam2, Y_prob, cam3, Y_prob, cam4, self.l_fea, loss

    def _generate_cam(self, features, attention_weights):
        batch_size = features.shape[0]
        A = attention_weights  # [batch_size, K, num_classes]
        cams = []
        for c in range(self.num_classes):
            # Exclude CLS token (first patch) for CAM generation
            cam = A[:, 1:, c].reshape(batch_size, 1, self.patch_grid, self.patch_grid)  # [batch_size, 1, 14, 14]
            cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=True)  # [batch_size, 1, 224, 224]
            cams.append(cam)
        cams = torch.stack(cams, dim=1).squeeze(2)  # [batch_size, num_classes, 224, 224]
        print(f"Generated CAM shape: {cams.shape}")
        return cams

    def generate_cam(self, x, label, h5_path=None, coords=None, img_name=None):
        features = self._extract_patched_features(x) if h5_path is None else None
        if features is None:
            # Load features from H5 file
            if isinstance(h5_path, (list, tuple)):
                features_list = []
                for path in h5_path:
                    try:
                        with h5py.File(path, 'r') as h5:
                            feat = torch.from_numpy(h5['features'][:]).float().to(next(self.parameters()).device)
                            if feat.dim() == 1 and feat.shape[0] == 1024:
                                feat = self.backbone_projection(feat.unsqueeze(0))
                                feat = feat.unsqueeze(1).expand(-1, self.K, -1)
                            elif feat.dim() == 2 and feat.shape[0] == 197 and feat.shape[1] == 1024:
                                feat = self.backbone_projection(feat)
                            elif feat.dim() == 2 and feat.shape[1] == 197 and feat.shape[0] == 1024:
                                feat = self.backbone_projection(feat.transpose(0, 1))
                            elif feat.dim() == 2 and feat.shape[1] == 512:
                                feat = feat.unsqueeze(1).expand(-1, self.K, -1)
                            else:
                                raise ValueError(f"Unexpected H5 features shape {feat.shape}")
                            features_list.append(feat)
                    except Exception as e:
                        print(f"Error loading H5 file {path}: {str(e)}. Using backbone features.")
                        features_list.append(self._extract_patched_features(x))
                features = torch.cat(features_list, dim=0)
            else:
                try:
                    with h5py.File(h5_path, 'r') as h5:
                        features = torch.from_numpy(h5['features'][:]).float().to(next(self.parameters()).device)
                        if features.dim() == 1 and features.shape[0] == 1024:
                            features = self.backbone_projection(features.unsqueeze(0))
                            features = features.unsqueeze(1).expand(-1, self.K, -1)
                        elif features.dim() == 2 and features.shape[0] == 197 and features.shape[1] == 1024:
                            features = self.backbone_projection(features)
                        elif features.dim() == 2 and features.shape[1] == 197 and features.shape[0] == 1024:
                            features = self.backbone_projection(features.transpose(0, 1))
                        elif features.dim() == 2 and features.shape[1] == 512:
                            features = features.unsqueeze(1).expand(-1, self.K, -1)
                        else:
                            raise ValueError(f"Unexpected H5 features shape {features.shape}")
                except Exception as e:
                    print(f"Error loading H5 file {h5_path}: {str(e)}. Using backbone features.")
                    features = self._extract_patched_features(x)

        A, _ = self.attention_net(features)
        A = F.softmax(A, dim=1)
        cam1 = self._generate_cam(features, A)
        return cam1, cam1, cam1, cam1, label

    def generate_segmentation(self, cams, coords, threshold=0.1):
        b, c, h, w = cams[0].shape
        cams = torch.stack(cams, dim=0)  # [num_cams, batch_size, num_classes, h, w]
        cams = cams.max(dim=0)[0]  # [batch_size, num_classes, h, w]
        
        cams = cams.cpu().data.numpy()
        cams = np.maximum(cams, 0)
        channel_max = np.max(cams, axis=(2, 3), keepdims=True)
        channel_min = np.min(cams, axis=(2, 3), keepdims=True)
        cams = (cams - channel_min) / (channel_max - channel_min + 1e-6)
        cams = torch.from_numpy(cams).float().to(next(self.parameters()).device)
        
        cams = F.interpolate(cams, size=(h, w), mode="bilinear", align_corners=True)
        cam_max = torch.max(cams, dim=1, keepdim=True)[0]
        bg_cam = (1 - cam_max) ** 10
        cam_all = torch.cat([cams, bg_cam], dim=1)  # [batch_size, num_classes + 1, h, w]

        # Generate class-wise binary masks
        seg = torch.zeros_like(cam_all)
        for c in range(self.num_classes):
            seg[:, c] = (cams[:, c] > threshold).float()
        seg[:, -1] = (bg_cam.squeeze(1) > threshold).float()  # Background as last channel
        print(f"segmentation shape (ViLa_MIL_Model): {seg.shape}")
        return seg[:, :self.num_classes], seg  # [batch_size, num_classes, h, w], [batch_size, num_classes + 1, h, w]
