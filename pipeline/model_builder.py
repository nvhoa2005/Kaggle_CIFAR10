import torch
from torch import nn
import torchvision.models as models

class EfficientNetB0(nn.Module):
    """Creates an EfficientNetB0 architecture for transfer learning.

    Args:
        output_shape: An integer indicating number of output classes.
        pretrained: Whether to use pretrained weights on ImageNet.
    """
    def __init__(self, in_features: int, output_shape: int, pretrained: bool = True) -> None:
        super().__init__()
        # Load EfficientNetB0 pretrained on ImageNet1K
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.model = models.efficientnet_b0(weights=weights)

        # Mở toàn bộ trọng số (unfreeze) để fine-tune full model
        for param in self.model.parameters():
            param.requires_grad = True

        # Thay classifier cuối cùng bằng lớp Linear mới
        self.model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2, inplace=True), 
            torch.nn.Linear(in_features=in_features, 
                            out_features=output_shape, # same number of output units as our number of classes
                            bias=True))

    def forward(self, x: torch.Tensor):
        return self.model(x)