# import torch
# import torch.nn as nn

# class PLIPProjector(nn.Module):
#     def __init__(self, local_model_path=None, device=None):
#         super(PLIPProjector, self).__init__()
#         self.D = 512  # Original feature dimension
#         self.reduced_dim = 128  # Reduced dimension after MLP
#         # Initialize layers with corrected input dimension for text_mlp
#         self.image_mlp = nn.Sequential(
#             nn.Linear(self.D, self.D // 2),  # Input: 512, Output: 256
#             nn.ReLU(),
#             nn.Linear(self.D // 2, self.reduced_dim),  # 256 -> 128
#             nn.ReLU(),
#             nn.Linear(self.reduced_dim, self.reduced_dim)  # 128 -> 128
#         )
#         self.text_mlp = nn.Sequential(
#             nn.Linear(self.D, self.D // 2),  # Input: 512, Output: 256
#             nn.ReLU(),
#             nn.Linear(self.D // 2, self.reduced_dim),  # 256 -> 128
#             nn.ReLU(),
#             nn.Linear(self.reduced_dim, self.reduced_dim)  # 128 -> 128
#         )
#         self.temperature = nn.Parameter(torch.ones([]) * 0.07)
#         if local_model_path:
#             print(f"Initialized PLIPProjector with local_model_path: {local_model_path}")
#         # Move to device if specified
#         if device is not None:
#             self.to(device)

#     def ImageMLP(self, x):
#         print(f"ImageMLP input shape: {x.shape}")
#         x = self.image_mlp(x)
#         print(f"ImageMLP output shape: {x.shape}")
#         return x

#     def TextMLP(self, x):
#         print(f"TextMLP input shape: {x.shape}")
#         # Handle 2D or 3D input
#         if x.dim() == 2:  # [num_classes, dim]
#             if x.shape[1] == self.D:  # Original dimension (512)
#                 x = self.text_mlp(x)
#             elif x.shape[1] == self.reduced_dim:  # Already reduced (128)
#                 x = x  # No further processing
#             else:
#                 raise ValueError(f"Unsupported input dimension size: {x.shape[1]}, expected {self.D} or {self.reduced_dim}")
#         elif x.dim() == 3:  # [num_classes, batch_size, dim]
#             num_classes, batch_size, dim = x.shape
#             if dim == self.D:  # Original dimension (512)
#                 x = x.permute(1, 0, 2).reshape(batch_size * num_classes, -1)  # [batch_size * num_classes, dim]
#                 x = self.text_mlp(x)
#                 x = x.reshape(batch_size, num_classes, -1).permute(1, 0, 2)  # [num_classes, batch_size, reduced_dim]
#             elif dim == self.reduced_dim:  # Already reduced (128)
#                 x = x  # No further processing
#             else:
#                 raise ValueError(f"Unsupported input dimension size: {dim}, expected {self.D} or {self.reduced_dim}")
#         else:
#             raise ValueError(f"Unsupported input dimension: {x.dim()}")
#         print(f"TextMLP output shape: {x.shape}")
#         return x

#     def forward(self, image_features, text_features):
#         print(f"Projector forward - image_features shape: {image_features.shape}, text_features shape: {text_features.shape}")
#         # Process image features
#         image_embeds = self.ImageMLP(image_features)  # [batch_size, num_classes, 128]
#         # Process text features only if not already reduced
#         if text_features.shape[-1] == self.D:  # [num_classes, batch_size, 512] or [num_classes, 512]
#             text_embeds = self.TextMLP(text_features)  # [num_classes, batch_size, 128]
#         else:  # Already reduced to 128
#             text_embeds = text_features  # [num_classes, batch_size, 128] or [num_classes, 128]
#             if text_embeds.dim() == 3:
#                 text_embeds = text_embeds.squeeze(1)  # [num_classes, batch_size, 128] -> [num_classes, 128] if batch_size=1
#             elif text_embeds.dim() == 2:
#                 text_embeds = text_embeds  # [num_classes, 128]
#             text_embeds = text_embeds.t()  # [128, num_classes]
#         # Normalize embeddings
#         image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True).clamp(min=1e-8)
#         text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True).clamp(min=1e-8)
#         # Einsum for similarity with corrected dimensions
#         print(f"image_embeds shape before einsum: {image_embeds.shape}")
#         print(f"text_embeds shape before einsum: {text_embeds.shape}")
#         logits = torch.einsum('bik,kj->bij', image_embeds, text_embeds) / self.temperature.exp()
#         print(f"Projector logits shape: {logits.shape}")
#         # Squeeze if the last dimension is 1
#         if logits.size(-1) == 1:
#             logits = logits.squeeze(-1)
#         print(f"Projector final logits shape: {logits.shape}")
#         return logits



import torch
import torch.nn as nn

class PLIPProjector(nn.Module):
    def __init__(self, local_model_path=None, device=None):
        super(PLIPProjector, self).__init__()
        self.D = 512  # Match ViLa_MIL_Model's feature dimension
        # Initialize layers without device, move to device later
        self.image_mlp = nn.Sequential(
            nn.Linear(self.D, self.D // 2),  # Input: 512, Output: 256
            nn.ReLU(),
            nn.Linear(self.D // 2, self.D // 4),  # 256 -> 128
            nn.ReLU(),
            nn.Linear(self.D // 4, self.D // 4)  # 128 -> 128
        )
        self.text_mlp = nn.Sequential(
            nn.Linear(self.D, self.D // 2),  # Input: 512, Output: 256
            nn.ReLU(),
            nn.Linear(self.D // 2, self.D // 4),  # 256 -> 128
            nn.ReLU(),
            nn.Linear(self.D // 4, self.D // 4)  # 128 -> 128
        )
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)
        if local_model_path:
            print(f"Initialized PLIPProjector with local_model_path: {local_model_path}")
        # Move to device if specified
        if device is not None:
            self.to(device)

    def ImageMLP(self, x):
        print(f"ImageMLP input shape: {x.shape}")
        x = self.image_mlp(x)
        print(f"ImageMLP output shape: {x.shape}")
        return x

    def TextMLP(self, x):
        print(f"TextMLP input shape: {x.shape}")
        x = self.text_mlp(x)
        print(f"TextMLP output shape: {x.shape}")
        return x

    def forward(self, image_features, text_features):
        print(f"Projector forward - image_features shape: {image_features.shape}, text_features shape: {text_features.shape}")
        image_embeds = self.ImageMLP(image_features)  # [batch, num_classes, 128]
        text_embeds = text_features  # [num_classes, 128], already processed by TextMLP in ViLa_MIL_Model
        # Fix: Avoid in-place normalization
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        logits = torch.einsum('bcd,cd->bc', image_embeds, text_embeds) / self.temperature
        print(f"Projector logits shape: {logits.shape}")
        return logits

