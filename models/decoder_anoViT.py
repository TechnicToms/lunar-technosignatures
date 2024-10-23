import torch
import torchvision
from torchsummary import summary


class AnoViTDecoder(torch.nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=2)
        self.bn1 = torch.nn.BatchNorm2d(num_features=64)
        
        self.conv2 = torch.nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2)
        self.bn2 = torch.nn.BatchNorm2d(num_features=64)
        
        self.conv3 = torch.nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2)
        self.bn3 = torch.nn.BatchNorm2d(num_features=64)
        
        self.conv4 = torch.nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2)
        self.bn4 = torch.nn.BatchNorm2d(num_features=64)
        
        self.conv5 = torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7)
        self.conv6 = torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7)
        self.conv7 = torch.nn.Conv2d(in_channels=16, out_channels=3, kernel_size=4)
        
        self.activation = torch.nn.ReLU()
        self.last_activation = torch.nn.Tanh()
         
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn1(self.conv1(x))
        x = self.activation(x)
        
        x = self.bn2(self.conv2(x))
        x = self.activation(x)
        
        x = self.bn3(self.conv3(x))
        x = self.activation(x)
        
        x = self.bn4(self.conv4(x))
        x = self.activation(x)
        
        x = self.activation(self.conv5(x))
        x = self.activation(self.conv6(x))
        x = self.last_activation(self.conv7(x))
        return x


if __name__ == '__main__':
    model = AnoViTDecoder(128)
    summary(model, input_size=(128, 14, 14), device="cpu")
    print('finished!')