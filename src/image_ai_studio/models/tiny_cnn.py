import torch
from torch import Tensor, nn


class TinyCNN(nn.Module):
    """Static CNN, no control flow: Conv-ReLU-Pool-Conv-ReLU-GAP-Linear.

    Input:  [1, 3, 224, 224] float32 NCHW
    Output: [1, 10] float32
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.gap(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    model = TinyCNN().eval()
    with torch.inference_mode():
        out = model(torch.randn(1, 3, 224, 224))
    print(out.shape)
