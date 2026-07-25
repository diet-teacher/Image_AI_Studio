import torch
from torch import Tensor, nn

from image_ai_studio.models.residual_block import ResidualBlock


class TinyResidualCNN(nn.Module):
    """Conv -> ResidualBlock -> GAP -> Linear.

    Exercises a skip connection (with BatchNorm running stats) in addition
    to the plain-Sequential path covered by TinyCNN, without any dynamic
    (input-value-dependent) control flow.

    Input:  [1, 3, 224, 224] float32 NCHW
    Output: [1, 10] float32
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.res_block = ResidualBlock(in_channels=16, out_channels=32, stride=2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.res_block(x)
        x = self.gap(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    model = TinyResidualCNN().eval()
    with torch.inference_mode():
        out = model(torch.randn(1, 3, 224, 224))
    print(out.shape)
