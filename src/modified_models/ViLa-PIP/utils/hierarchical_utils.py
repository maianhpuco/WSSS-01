import torch
import torch.nn.functional as F
import torch.nn as nn

class TextFeatureProjector(nn.Module):
    def __init__(self, in_dim=512, out_dim=128, device=None):
        super().__init__()
        self.projector = nn.Linear(in_dim, out_dim).to(device if device is not None else torch.device('cpu'))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.projector(x)

def pair_features(fg_features, bg_features, text_features, labels, device=None):
    """
    Pair foreground and background features with corresponding text embeddings for contrastive loss.
    Args:
        fg_features (torch.Tensor): Foreground features [num_active_classes, 1, 128].
        bg_features (torch.Tensor): Background features [num_active_classes, 1, 128].
        text_features (torch.Tensor): Text embeddings from ViLa_MIL_Model [num_classes, 512].
        labels (torch.Tensor): Binary labels of shape (N, C) indicating positive classes.
        device (torch.device): Device to perform computations on.
    Returns:
        dict: Dictionary containing paired features and text embeddings.
    """
    text_projector = TextFeatureProjector(in_dim=512, out_dim=128, device=device)
    
    print(f"pair_features - text_features shape: {text_features.shape}")
    text_features_projected = text_projector(text_features.clone().detach())
    print(f"pair_features - text_features_projected shape: {text_features_projected.shape}")

    batch_indices, class_indices = torch.where(labels > 0.01)
    print(f"pair_features - num foreground samples: {len(batch_indices)}")

    if len(batch_indices) == 0:
        return {
            'fg_features': torch.tensor([], device=device),
            'bg_features': torch.tensor([], device=device),
            'fg_text': torch.tensor([], device=device),
            'bg_text': torch.tensor([], device=device)
        }

    max_samples = min(len(batch_indices), 8)
    batch_indices = batch_indices[:max_samples]
    class_indices = class_indices[:max_samples]

    paired_fg_features = []
    paired_bg_features = []
    paired_fg_text = []
    paired_bg_text = []

    for i in range(max_samples):
        curr_idx = batch_indices[i]
        curr_class = class_indices[i]
        curr_class = min(curr_class, text_features_projected.size(0) - 1)

        curr_fg = fg_features[i].squeeze(1).clone() if fg_features.dim() == 3 and i < fg_features.shape[0] else torch.zeros(128, device=device)
        curr_bg = bg_features[i].squeeze(1).clone() if bg_features.dim() == 3 and i < bg_features.shape[0] else torch.zeros(128, device=device)

        curr_fg_text = text_features_projected[curr_class].clone().detach()
        bg_text_indices = [j for j in range(text_features_projected.size(0)) if j != curr_class]
        if bg_text_indices:
            curr_bg_text = text_features_projected[bg_text_indices].mean(dim=0).clone().detach()
        else:
            curr_bg_text = torch.zeros_like(curr_fg)

        paired_fg_features.append(curr_fg)
        paired_bg_features.append(curr_bg)
        paired_fg_text.append(curr_fg_text)
        paired_bg_text.append(curr_bg_text)

    paired_fg_features = torch.stack(paired_fg_features) if paired_fg_features else torch.tensor([], device=device)
    paired_bg_features = torch.stack(paired_bg_features) if paired_bg_features else torch.tensor([], device=device)
    paired_fg_text = torch.stack(paired_fg_text) if paired_fg_text else torch.tensor([], device=device)
    paired_bg_text = torch.stack(paired_bg_text) if paired_bg_text else torch.tensor([], device=device)

    print(f"pair_features - paired_fg_features shape: {paired_fg_features.shape}, paired_fg_text shape: {paired_fg_text.shape}")
    print(f"pair_features - paired_bg_features shape: {paired_bg_features.shape}, paired_bg_text shape: {paired_bg_text.shape}")

    return {
        'fg_features': paired_fg_features,
        'bg_features': paired_bg_features,
        'fg_text': paired_fg_text,
        'bg_text': paired_bg_text
    }

def merge_to_parent_predictions(predictions, attention_weights, method='attention'):
    """
    Merge predictions using attention weights from ViLa_MIL_Model.
    
    Args:
        predictions (torch.Tensor): Predictions from the model, shape [batch_size, num_classes]
        attention_weights (torch.Tensor): Attention weights from Attn_Net_Gated, shape [batch_size, num_classes, K]
        method (str): Aggregation method ('attention' uses attention weights)
    
    Returns:
        torch.Tensor: Merged parent predictions, shape [batch_size, num_classes]
    """
    if method != 'attention':
        raise ValueError("Only 'attention' method is supported for this model.")
    
    batch_size, num_classes = predictions.shape
    # Ensure attention_weights are normalized
    attention_weights = F.softmax(attention_weights, dim=2)  # [batch_size, num_classes, K]
    
    # Aggregate predictions using attention weights
    parent_preds = predictions.clone()  # Base predictions
    for b in range(batch_size):
        for c in range(num_classes):
            attn_w = attention_weights[b, c]  # [K]
            patch_contrib = torch.sum(attn_w)  # Normalize attention contribution
            if patch_contrib > 0:
                parent_preds[b, c] = predictions[b, c] * (patch_contrib / attention_weights.shape[2])
    
    return parent_preds

def merge_subclass_cams_to_parent(cams, attention_weights, method='attention'):
    """
    Merge CAMs using attention weights from ViLa_MIL_Model.
    
    Args:
        cams (torch.Tensor): CAMs from the model, shape [batch_size, num_classes, H, W]
        attention_weights (torch.Tensor): Attention weights from Attn_Net_Gated, shape [batch_size, num_classes, K]
        method (str): Aggregation method ('attention' uses attention weights)
    
    Returns:
        torch.Tensor: Merged parent CAMs, shape [batch_size, num_classes, H, W]
    """
    if method != 'attention':
        raise ValueError("Only 'attention' method is supported for this model.")
    
    batch_size, num_classes, H, W = cams.shape
    parent_cams = torch.zeros_like(cams)
    
    # Normalize attention weights
    attention_weights = F.softmax(attention_weights, dim=2)  # [batch_size, num_classes, K]
    
    # Debug: Print shapes for verification
    print(f"merge_subclass_cams_to_parent - cams shape: {cams.shape}")
    print(f"merge_subclass_cams_to_parent - attention_weights shape: {attention_weights.shape}")
    
    # Compute the total number of pixels
    total_pixels = H * W  # 56 * 56 = 3136
    K = attention_weights.shape[2]  # 197 patches
    
    # Reshape attention weights to [batch_size, num_classes, K, 1] for interpolation per class
    attn_weights_reshaped = attention_weights.permute(0, 2, 1).unsqueeze(-1)  # [batch_size, K, num_classes, 1]
    
    # Interpolate attention weights to [H, W] for each class using bilinear mode
    attn_weights_upsampled = F.interpolate(
        attn_weights_reshaped,
        size=(H, W),
        mode='bilinear',
        align_corners=True
    )  # [batch_size, K, num_classes, H, W]
    
    # Aggregate over K dimension to get [batch_size, num_classes, H, W]
    attn_weights_aggregated = attn_weights_upsampled.mean(dim=1)  # [batch_size, num_classes, H, W]
    attn_weights_aggregated = F.softmax(attn_weights_aggregated, dim=1)  # Normalize across classes
    
    # Flatten CAMs for element-wise multiplication
    cams_flat = cams.view(batch_size, num_classes, -1)  # [batch_size, num_classes, H*W]
    
    # Debug: Verify shapes before multiplication
    print(f"merge_subclass_cams_to_parent - cams_flat shape: {cams_flat.shape}")
    print(f"merge_subclass_cams_to_parent - attn_weights_aggregated shape: {attn_weights_aggregated.shape}")
    
    # Apply attention weights directly to CAMs
    for b in range(batch_size):
        for c in range(num_classes):
            attn_w = attn_weights_aggregated[b, c]  # [H, W]
            cam_flat = cams_flat[b, c]  # [H*W]
            # Ensure shapes match by flattening attn_w
            attn_w_flat = attn_w.view(-1)  # [H*W]
            if attn_w_flat.size(0) != cam_flat.size(0):
                print(f"Mismatch detected: attn_w_flat size {attn_w_flat.size(0)}, cam_flat size {cam_flat.size(0)}")
                # Fallback: Repeat attn_w_flat to match cam_flat size with proper padding
                repeat_factor = (cam_flat.size(0) + attn_w_flat.size(0) - 1) // attn_w_flat.size(0)
                attn_w_flat = attn_w_flat.repeat(repeat_factor)[:cam_flat.size(0)]
                print(f"Adjusted attn_w_flat size: {attn_w_flat.size(0)}")
            parent_cam_flat = cam_flat * attn_w_flat  # Weighted sum
            parent_cams[b, c] = parent_cam_flat.view(H, W)
    
    print(f"merge_subclass_cams_to_parent - parent_cams shape: {parent_cams.shape}")
    return parent_cams

def expand_parent_to_subclass_labels(parent_labels, _):
    """
    Expand parent labels to subclass labels (no expansion needed for this model).
    
    Args:
        parent_labels (torch.Tensor): Parent labels, shape [batch_size, num_classes]
        _ (any): Placeholder for k_list (ignored)
    
    Returns:
        torch.Tensor: Same as parent_labels, shape [batch_size, num_classes]
    """
    return parent_labels


