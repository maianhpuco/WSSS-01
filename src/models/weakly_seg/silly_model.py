import torch.nn as nn

class TokenCAMCNN(nn.Module):
	def __init__(self, num_classes=4):
		super().__init__()
		self.features = nn.Sequential(
			nn.Conv2d(1, 32, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(32, 64, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 128, kernel_size=3, padding=1),
			nn.ReLU(inplace=True)
		)
		self.classifier = nn.Conv2d(128, num_classes, kernel_size=1)

	def forward(self, x):
		feat = self.features(x)      # [B, 128, 40, 40]
		logits = self.classifier(feat)  # [B, num_classes, 40, 40]
		return logits, feat
