import torch
from enum import Enum


class EfficientAdModelSize(str, Enum):
    """Supported EfficientAd model sizes"""

    M = "medium"
    S = "small"


class PDNSmall(torch.nn.Module):
    """Patch description network (PDN). EfficientAD-S
    """
    def __init__(self, out_channels: int, padding: bool = False) -> None:
        """Small patch description network for EfficientAD Algorithm.

        Args:
            out_channels (int): Number of output channels of Network.
            padding (bool, optional): if padding should be used. Defaults to False.
        """
        super().__init__()
        pad_mult = 1 if padding else 0
        
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=128, kernel_size=4, stride=1, padding=3 * pad_mult)
        self.conv2 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=1, padding=3 * pad_mult)
        self.conv3 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1 * pad_mult)
        self.conv4 = torch.nn.Conv2d(in_channels=256, out_channels=out_channels, kernel_size=4, stride=1, padding=0 * pad_mult)
        
        self.avgPool = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult)
        
        self.activation = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.conv1(x))
        x = self.avgPool(x)
        
        x = self.activation(self.conv2(x))
        x = self.avgPool(x)
        
        x = self.activation(self.conv3(x))
        x = self.conv4(x)

        return x


class PDNMedium(torch.nn.Module):
    """Patch description network (PDN). EfficientAD-M
    """
    def __init__(self, out_channels: int, padding: bool = False) -> None:
        """Medium patch description network for EfficientAD Algorithm.

        Args:
            out_channels (int): Number of output channels of Network.
            padding (bool, optional): if padding should be used. Defaults to False.
        """
        super().__init__()
        pad_mult = 1 if padding else 0
        
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=256, kernel_size=4, stride=1, padding=3 * pad_mult)
        self.conv2 = torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=3 * pad_mult)
        self.conv3 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0 * pad_mult)
        self.conv4 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1 * pad_mult)
        self.conv5 = torch.nn.Conv2d(in_channels=512, out_channels=out_channels, kernel_size=4, stride=1, padding=0 * pad_mult)
        self.conv6 = torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0 * pad_mult)
        
        self.avgPool = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult)
        
        self.activation = torch.nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.conv1(x))
        x = self.avgPool(x)
        
        x = self.activation(self.conv2(x))
        x = self.avgPool(x)
        
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))
        x = self.activation(self.conv5(x))
        x = self.conv6(x)

        return x

