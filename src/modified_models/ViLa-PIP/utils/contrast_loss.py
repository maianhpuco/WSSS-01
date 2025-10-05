import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELossFG(nn.Module):
    def __init__(self, temperature=0.07):
        super(InfoNCELossFG, self).__init__()
        self.temperature = temperature

    def forward(self, fg_img_feature, fg_pro_feature, bg_pro_feature):
        # Debug shapes
        print(f"InfoNCELossFG - fg_img_feature shape: {fg_img_feature.shape}, fg_pro_feature shape: {fg_pro_feature.shape}, bg_pro_feature shape: {bg_pro_feature.shape}")

        # Ensure fg_img_feature has shape [N, 1, D] and fg_pro_feature/bg_pro_feature have shape [N, D]
        N = fg_img_feature.size(0)  # Number of samples
        D = fg_img_feature.size(2)  # Feature dimension

        # Reshape fg_img_feature to [N, D] for einsum compatibility
        fg_img_feature = fg_img_feature.squeeze(1)  # [N, D], e.g., [4, 128]

        # Compute positive logits (similarity with corresponding positive text feature)
        pos_logits = torch.sum(fg_img_feature * fg_pro_feature, dim=1, keepdim=True) / self.temperature  # [N, 1]
        print(f"InfoNCELossFG - pos_logits shape: {pos_logits.shape}")

        # Compute negative logits (similarity with background text features)
        neg_logits = torch.matmul(fg_img_feature, bg_pro_feature.T) / self.temperature  # [N, N]
        print(f"InfoNCELossFG - neg_logits shape: {neg_logits.shape}")

        # Concatenate positive and negative logits
        logits = torch.cat([pos_logits, neg_logits], dim=1)  # [N, N+1]
        print(f"InfoNCELossFG - logits shape: {logits.shape}")

        # Prepare labels for InfoNCE loss (first column is positive)
        labels = torch.zeros(N, dtype=torch.long, device=fg_img_feature.device)  # [N], all point to the first column (positive)

        # Compute cross-entropy loss
        loss = F.cross_entropy(logits, labels)
        print(f"InfoNCELossFG - loss: {loss.item()}")
        return loss

class InfoNCELossBG(nn.Module):
    def __init__(self, temperature=0.07):
        super(InfoNCELossBG, self).__init__()
        self.temperature = temperature

    def forward(self, bg_img_feature, fg_pro_feature, bg_pro_feature):
        # Debug shapes
        print(f"InfoNCELossBG - bg_img_feature shape: {bg_img_feature.shape}, fg_pro_feature shape: {fg_pro_feature.shape}, bg_pro_feature shape: {bg_pro_feature.shape}")

        # Ensure bg_img_feature has shape [N, 1, D] and fg_pro_feature/bg_pro_feature have shape [N, D]
        N = bg_img_feature.size(0)
        D = bg_img_feature.size(2)

        # Reshape bg_img_feature to [N, D] for einsum compatibility
        bg_img_feature = bg_img_feature.squeeze(1)  # [N, D]

        # Compute positive logits (similarity with corresponding positive text feature)
        pos_logits = torch.sum(bg_img_feature * bg_pro_feature, dim=1, keepdim=True) / self.temperature  # [N, 1]
        print(f"InfoNCELossBG - pos_logits shape: {pos_logits.shape}")

        # Compute negative logits (similarity with foreground text features)
        neg_logits = torch.matmul(bg_img_feature, fg_pro_feature.T) / self.temperature  # [N, N]
        print(f"InfoNCELossBG - neg_logits shape: {neg_logits.shape}")

        # Concatenate positive and negative logits
        logits = torch.cat([pos_logits, neg_logits], dim=1)  # [N, N+1]
        print(f"InfoNCELossBG - logits shape: {logits.shape}")

        # Prepare labels for InfoNCE loss (first column is positive)
        labels = torch.zeros(N, dtype=torch.long, device=bg_img_feature.device)  # [N], all point to the first column (positive)

        # Compute cross-entropy loss
        loss = F.cross_entropy(logits, labels)
        print(f"InfoNCELossBG - loss: {loss.item()}")
        return loss
    