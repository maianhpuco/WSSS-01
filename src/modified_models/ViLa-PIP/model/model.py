from model.model_ViLa_MIL import ViLa_MIL_Model, PromptLearner, TextEncoder
import pickle as pkl
from functools import partial
from omegaconf import OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_
from model.projector import PLIPProjector
import open_clip
import h5py
import numpy as np
import sys
import os
ROOT_FOLDER = r"D:\NghienCuu\NghienCuuPaper\Source_Code\WSSS-01\src\modified_externals\ViLa-PIP"
sys.path.insert(0, ROOT_FOLDER)
from utils.hierarchical_utils import merge_to_parent_predictions, merge_subclass_cams_to_parent, pair_features

class AdaptiveLayer(nn.Module):
    def __init__(self, in_dim, n_ratio, out_dim):
        super().__init__()
        hidden_dim = int(in_dim * n_ratio)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class ClsNetwork(nn.Module):
    def __init__(self,
                 backbone='vit_b_16',
                 cls_num_classes=4,
                 stride=[4, 2, 2, 1],
                 pretrained=True,
                 n_ratio=0.5,
                 l_fea_path=None,
                 text_prompt=None):
        super().__init__()
        self.cls_num_classes = cls_num_classes
        self.total_classes = cls_num_classes
        self.stride = stride
        self.n_ratio = n_ratio
        self.pretrained = pretrained
        self.text_prompt = text_prompt if text_prompt is not None else [
            "A WSI of Tumor with visually descriptive characteristics of irregularly shaped regions, dense cellularity, heterogeneous staining, and distortion of adjacent structures due to growth, as well as atypical cells, enlarged nuclei, prominent nucleoli, high nuclear-to-cytoplasmic ratio, and mitotic figures.",
            "A WSI of Stroma with visually descriptive characteristics of fibrous connective tissue, lighter staining, low cellularity, and surrounding or infiltrating tumor areas, as well as elongated fibroblasts, collagen bundles, eosinophilic matrix, blood vessels, and occasional inflammatory cells.",
            "A WSI of Lymphocyte with visually descriptive characteristics of small dark clusters or infiltrates, often at tumor-stroma interfaces, appearing as speckled blue-purple areas, as well as small round cells, hyperchromatic nuclei, scant cytoplasm, and clustering in immune responses.",
            "A WSI of Necrosis with visually descriptive characteristics of pale amorphous zones, loss of structure, hypoeosinophilic appearance, and contrast with viable tissue, as well as cellular debris, karyorrhectic nuclei, cytoplasmic remnants, and infiltration by inflammatory cells."
        ]
        self.backbone = ViLa_MIL_Model(
            config=OmegaConf.create({
                "prototype_number": 3,
                "input_size": [224, 224],
                "text_prompt": self.text_prompt,
                "n_ratio": self.n_ratio
            }),
            num_classes=self.cls_num_classes
        )
        self.D = self.backbone.D
        self.prototype_number = self.backbone.prototype_number
        self.learnable_image_center = self.backbone.learnable_image_center
        clip_model, _, _ = open_clip.create_model_and_transforms(
            "RN50", pretrained='openai', quick_gelu=True)
        self.prompt_learner = PromptLearner(
            self.text_prompt, clip_model.float())
        self.text_encoder = TextEncoder(clip_model.float())
        self.text_projection = nn.Linear(512, 512).float()
        self.projector = PLIPProjector(
            local_model_path=r"D:\NghienCuu\NghienCuuPaper\Source_Code\WSSS-01\pretrained_models\vinid_plip")
        self.pooling = F.adaptive_avg_pool2d
        self.l_fc1 = AdaptiveLayer(512, n_ratio, self.D)
        self.l_fc2 = AdaptiveLayer(512, n_ratio, self.D)
        self.l_fc3 = AdaptiveLayer(512, n_ratio, self.D)
        self.l_fc4 = AdaptiveLayer(512, n_ratio, self.D)
        text_features = self.text_encoder(self.prompt_learner(),
                                         self.prompt_learner.tokenized_prompts)
        text_features = self.text_projection(text_features)
        print(f"Initialized l_fea shape: {text_features.shape}")
        if text_features.shape[0] != self.total_classes:
            raise ValueError(f"Expected {self.total_classes} text features, got {text_features.shape[0]}")
        self.register_buffer("l_fea", text_features)
        self.register_buffer("default_coords", None)
        self.logit_scale1 = nn.Parameter(torch.ones([1]) * 1 / 0.07)
        self.logit_scale2 = nn.Parameter(torch.ones([1]) * 1 / 0.07)
        self.logit_scale3 = nn.Parameter(torch.ones([1]) * 1 / 0.07)
        self.logit_scale4 = nn.Parameter(torch.ones([1]) * 1 / 0.07)
        self.proj1 = nn.Linear(64, 512)
        self.proj2 = nn.Linear(64, 512)
        self.proj3 = nn.Linear(256, 512)
        self.proj4 = nn.Linear(256, 512)
        self.apply(self._init_weights)
        self.loss_ce = nn.CrossEntropyLoss()
        self.loss_bce = nn.BCEWithLogitsLoss()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_param_groups(self):
        regularized = []
        not_regularized = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.endswith(".bias") or len(param.shape) == 1:
                not_regularized.append(param)
            else:
                regularized.append(param)
        return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]

    def forward(self, x, labels=None, fg_bg_labels=None, h5_path=None, coords=None, img_name=None):
        batch_size = x.shape[0]
        device = next(self.parameters()).device
        text_prompts = self.prompt_learner()
        print(f"text_prompts shape (ClsNetwork): {text_prompts.shape}")
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        text_features = self.text_encoder(text_prompts, tokenized_prompts)
        print(f"text_features before projection shape (ClsNetwork): {text_features.shape}")
        text_features = self.text_projection(text_features.clone())
        print(f"text_features after projection shape (ClsNetwork): {text_features.shape}")
        text_features_4 = self.projector.TextMLP(text_features[:self.cls_num_classes].clone())
        print(f"text_features after TextMLP shape (ClsNetwork): {text_features_4.shape}")
        l_fea = self.l_fea.clone().detach()
        print(f"l_fea shape (ClsNetwork): {l_fea.shape}")
        coords = coords if coords is not None else self.default_coords
        
        # Call backbone and unpack 10 outputs
        cls1, cam1, cls2, cam2, cls3, cam3, cls4, cam4, l_fea_backbone, backbone_loss = self.backbone(
            x, labels=labels, fg_bg_labels=fg_bg_labels, h5_path=h5_path, coords=coords, img_name=img_name
        )
        print(f"Backbone output shapes - cls1: {cls1.shape}, cam1: {cam1.shape}, l_fea_backbone: {l_fea_backbone.shape}")
        
        # Map to 5 outputs for training
        Y_prob = (cls1 + cls2 + cls3 + cls4) / 4  # Average the classifier outputs
        Y_hat = torch.argmax(Y_prob, dim=1)
        fg_bg_prob = torch.sigmoid(self.backbone.fg_bg_classifier(l_fea_backbone.mean(dim=0).unsqueeze(0)))
        attention_weights = self.backbone.attention_net(self.backbone.features)[0]
        
        # Additional processing for cls1, cam1, ..., cls4, cam4
        M = self.backbone.features
        print(f"M shape before processing (ClsNetwork): {M.shape}")
        if M.shape[1] == 1:
            M_processed = M
            num_patches = 1
        else:
            M_processed = M[:, 1:, :]  # Exclude CLS token
            num_patches = M_processed.size(1)  # Dynamically set num_patches
        print(f"M_processed shape before operations: {M_processed.shape}")
        M_processed = M_processed.contiguous()
        print(f"M_processed shape before bmm: {M_processed.shape}")
        _x_flat = M_processed.reshape(batch_size * num_patches, -1)
        print(f"_x_flat shape (ClsNetwork): {_x_flat.shape}")
        _x1 = self.l_fc1(_x_flat)
        _x2 = self.l_fc2(_x_flat)
        _x3 = self.l_fc3(_x_flat)
        _x4 = self.l_fc4(_x_flat)
        if num_patches == 196:
            _x1 = _x1.reshape(batch_size, 14, 14, 512).permute(0, 3, 1, 2)
            _x2 = _x2.reshape(batch_size, 14, 14, 512).permute(0, 3, 1, 2)
            _x3 = _x3.reshape(batch_size, 14, 14, 512).permute(0, 3, 1, 2)
            _x4 = _x4.reshape(batch_size, 14, 14, 512).permute(0, 3, 1, 2)
        else:
            _x1 = _x1.reshape(batch_size, num_patches, 1, 512).permute(0, 3, 1, 2)
            _x2 = _x2.reshape(batch_size, num_patches, 1, 512).permute(0, 3, 1, 2)
            _x3 = _x3.reshape(batch_size, num_patches, 1, 512).permute(0, 3, 1, 2)
            _x4 = _x4.reshape(batch_size, num_patches, 1, 512).permute(0, 3, 1, 2)
        _x1 = F.interpolate(_x1, size=(56, 56), mode='bilinear', align_corners=False)
        _x2 = F.interpolate(_x2, size=(56, 56), mode='bilinear', align_corners=False)
        _x3 = F.interpolate(_x3, size=(56, 56), mode='bilinear', align_corners=False)
        _x4 = F.interpolate(_x4, size=(56, 56), mode='bilinear', align_corners=False)
        _x1_spatial = _x1.permute(0, 2, 3, 1).reshape(-1, 512)
        _x2_spatial = _x2.permute(0, 2, 3, 1).reshape(-1, 512)
        _x3_spatial = _x3.permute(0, 2, 3, 1).reshape(-1, 512)
        _x4_spatial = _x4.permute(0, 2, 3, 1).reshape(-1, 512)
        _x1_projected = self.logit_scale1 * (_x1_spatial @ l_fea.clone().detach().t())
        cam1 = _x1_projected.reshape(batch_size, 56, 56, self.total_classes).permute(0, 3, 1, 2)
        cls1 = self.pooling(cam1, (1, 1)).reshape(-1, self.total_classes)
        print(f"cls1 shape before merging: {cls1.shape}")
        _x2_projected = self.logit_scale2 * (_x2_spatial @ l_fea.clone().detach().t())
        cam2 = _x2_projected.reshape(batch_size, 56, 56, self.total_classes).permute(0, 3, 1, 2)
        cls2 = self.pooling(cam2, (1, 1)).reshape(-1, self.total_classes)
        print(f"cls2 shape before merging: {cls2.shape}")
        _x3_projected = self.logit_scale3 * (_x3_spatial @ l_fea.clone().detach().t())
        cam3 = _x3_projected.reshape(batch_size, 56, 56, self.total_classes).permute(0, 3, 1, 2)
        cls3 = self.pooling(cam3, (1, 1)).reshape(-1, self.total_classes)
        print(f"cls3 shape before merging: {cls3.shape}")
        _x4_projected = self.logit_scale4 * (_x4_spatial @ l_fea.clone().detach().t())
        cam4 = _x4_projected.reshape(batch_size, 56, 56, self.total_classes).permute(0, 3, 1, 2)
        cls4 = self.pooling(cam4, (1, 1)).reshape(-1, self.total_classes)
        print(f"cls4 shape before merging: {cls4.shape}")
        cls1 = merge_to_parent_predictions(cls1, attention_weights[:, 1:, :], method='attention')
        cls2 = merge_to_parent_predictions(cls2, attention_weights[:, 1:, :], method='attention')
        cls3 = merge_to_parent_predictions(cls3, attention_weights[:, 1:, :], method='attention')
        cls4 = merge_to_parent_predictions(cls4, attention_weights[:, 1:, :], method='attention')
        print(f"cls1 shape after merging: {cls1.shape}")
        print(f"cls2 shape after merging: {cls2.shape}")
        print(f"cls3 shape after merging: {cls3.shape}")
        print(f"cls4 shape after merging: {cls4.shape}")
        cam1 = merge_subclass_cams_to_parent(cam1, attention_weights[:, 1:, :], method='attention')
        cam2 = merge_subclass_cams_to_parent(cam2, attention_weights[:, 1:, :], method='attention')
        cam3 = merge_subclass_cams_to_parent(cam3, attention_weights[:, 1:, :], method='attention')
        cam4 = merge_subclass_cams_to_parent(cam4, attention_weights[:, 1:, :], method='attention')
        print(f"cam1 shape after merging: {cam1.shape}")
        print(f"cam2 shape after merging: {cam2.shape}")
        print(f"cam3 shape after merging: {cam3.shape}")
        print(f"cam4 shape after merging: {cam4.shape}")
        cam1 = cam1.squeeze(1) if cam1.dim() > 4 else cam1
        cam2 = cam2.squeeze(1) if cam2.dim() > 4 else cam2
        cam3 = cam3.squeeze(1) if cam3.dim() > 4 else cam3
        cam4 = cam4.squeeze(1) if cam4.dim() > 4 else cam4
        print(f"cam1 shape after squeeze: {cam1.shape}")
        print(f"cam2 shape after squeeze: {cam2.shape}")
        print(f"cam3 shape after squeeze: {cam3.shape}")
        print(f"cam4 shape after squeeze: {cam4.shape}")
        compents_list = []
        for i in range(self.prototype_number):
            prototype = self.learnable_image_center[i:i + 1].expand(batch_size, -1, -1).to(device)
            compents, _ = self.backbone.cross_attention_1(prototype, M, M)
            compents = self.backbone.norm(compents + prototype)
            compents_list.append(compents)
        compents = torch.mean(torch.stack(compents_list, dim=0), dim=0)
        H = compents
        attention_weights_spatial = attention_weights[:, 1:, :]  # Exclude CLS token
        print(f"Attention weights spatial shape before permute: {attention_weights_spatial.shape}")
        num_patches = M_processed.size(1)
        if M_processed.dim() != 3 or M_processed.shape[1] != attention_weights_spatial.shape[1]:
            raise ValueError(f"M_processed shape {M_processed.shape} does not match expected [batch_size, {attention_weights_spatial.shape[1]}, {M_processed.shape[2]}] or attention_weights_spatial shape {attention_weights_spatial.shape}")
        image_features = torch.bmm(attention_weights_spatial.permute(0, 2, 1), M_processed)
        print(f"image_features shape: {image_features.shape}")
        image_context = torch.cat((H, M_processed), dim=1)
        text_context_input = text_features.unsqueeze(1).expand(batch_size, -1, -1, -1)
        image_context_expanded = image_context.repeat_interleave(self.cls_num_classes, dim=0)
        text_context_features, _ = self.backbone.cross_attention_2(
            text_context_input.reshape(batch_size * self.cls_num_classes, 1, self.D),
            image_context_expanded,
            image_context_expanded)
        text_context_features = text_context_features.reshape(batch_size, self.cls_num_classes, 1, self.D)
        # text_feature của 4 class = text_features_4 + text_feature của 3 class còn lại
        text_features_4 = text_context_features.squeeze(2).mean(dim=0) + text_features.clone().detach()
        text_features_4 = self.projector.TextMLP(text_features_4)
        logits = self.projector.forward(image_features, text_features_4)
        
        if labels is not None:
            labels_indices = labels.argmax(dim=1).to(x.device) if labels.dim() == 2 else labels
        else:
            labels_indices = torch.zeros(batch_size, dtype=torch.long, device=device)
        ce_loss = self.loss_ce(logits, labels_indices) + self.loss_ce(cls1, labels_indices) + \
                  self.loss_ce(cls2, labels_indices) + self.loss_ce(cls3, labels_indices) + \
                  self.loss_ce(cls4, labels_indices)
        if fg_bg_labels is not None:
            bce_loss = self.loss_bce(fg_bg_prob, fg_bg_labels.float())
        else:
            bce_loss = self.loss_bce(fg_bg_prob, torch.zeros(batch_size, 1, device=device))
        loss = ce_loss + bce_loss
        Y_prob = (cls1 + cls2 + cls3 + cls4) / 4
        print(f"Y_prob shape (ClsNetwork): {Y_prob.shape}")
        Y_hat = torch.argmax(Y_prob, dim=1)
        
        return cls1, cam1, cls2, cam2, cls3, cam3, cls4, cam4, l_fea, loss

    def generate_cam(self, x, label, h5_path=None, coords=None, fg_bg_labels=None, img_name=None):
        cls1, cam1, cls2, cam2, cls3, cam3, cls4, cam4, l_fea, loss = self.forward(
            x, labels=label, fg_bg_labels=fg_bg_labels, h5_path=h5_path, coords=coords, img_name=img_name
        )
        return cam1, cam2, cam3, cam4, loss

    def generate_segmentation(self, cams, coords, threshold=0.5):
        if isinstance(cams, torch.Tensor):
            cams = [cams] * 4
        cam = cams[0] + cams[1] + cams[2] + cams[3]  # [batch_size, num_classes, 56, 56]
        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)  # [batch_size, num_classes, 224, 224]
        seg = (cam > threshold).float()
        print(f"segmentation shape (ClsNetwork): {seg.shape}")
        return seg, seg

