import torch
import torch.nn.functional as F

from torchsummary import summary


class encoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.EncConv1 = torch.nn.Conv2d(in_channels=3,  out_channels=32, kernel_size=4, stride=2, padding=1)
        self.EncConv2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.EncConv3 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.EncConv4 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.EncConv5 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.EncConv6 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=8, stride=1, padding=0)
                
        self.activation = torch.nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.EncConv1(x))
        x = self.activation(self.EncConv2(x))
        
        x = self.activation(self.EncConv3(x))
        x = self.activation(self.EncConv4(x))
        
        x = self.activation(self.EncConv5(x))
        x = self.EncConv6(x)
        
        return x
    

class decoder(torch.nn.Module):
    def __init__(self, out_channels: int, padding: bool, img_size: tuple, *args, **kwargs) -> None:
        """Decoder element of the autoencoder

        Args:
            out_channels (int): Number of output channels
            padding (bool): wether to use padding or not
            img_size (tuple): size of input images
        """
        super().__init__(*args, **kwargs)
        self.img_size = img_size
        self.last_upsample = (
            int(img_size[0] / 4) if padding else int(img_size[0] / 4) - 8,
            int(img_size[1] / 4) if padding else int(img_size[1] / 4) - 8,
        )
        
        self.deconv2 = torch.nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv3 = torch.nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv1 = torch.nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv4 = torch.nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv5 = torch.nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv6 = torch.nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv7 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.deconv8 = torch.nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1)
        
        self.dropout1 = torch.nn.Dropout(p=0.2)
        self.dropout2 = torch.nn.Dropout(p=0.2)
        self.dropout3 = torch.nn.Dropout(p=0.2)
        self.dropout4 = torch.nn.Dropout(p=0.2)
        self.dropout5 = torch.nn.Dropout(p=0.2)
        self.dropout6 = torch.nn.Dropout(p=0.2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=(int(self.img_size[0] / 64) - 1, int(self.img_size[1] / 64) - 1), mode="bilinear", antialias=True)
        x = F.relu(self.deconv1(x))
        x = self.dropout1(x)
        
        x = F.interpolate(x, size=(int(self.img_size[0] / 32), int(self.img_size[1] / 32)), mode="bilinear", antialias=True)
        x = F.relu(self.deconv2(x))
        x = self.dropout2(x)
        
        x = F.interpolate(x, size=(int(self.img_size[0] / 16) - 1, int(self.img_size[1] / 16) - 1), mode="bilinear", antialias=True)
        x = F.relu(self.deconv3(x))
        x = self.dropout3(x)
        
        x = F.interpolate(x, size=(int(self.img_size[0] / 8), int(self.img_size[1] / 8)), mode="bilinear", antialias=True)
        x = F.relu(self.deconv4(x))
        x = self.dropout4(x)
        
        x = F.interpolate(x, size=(int(self.img_size[0] / 4) - 1, int(self.img_size[1] / 4) - 1), mode="bilinear", antialias=True)
        x = F.relu(self.deconv5(x))
        x = self.dropout5(x)
        
        x = F.interpolate(x, size=(int(self.img_size[0] / 2) - 1, int(self.img_size[1] / 2) - 1), mode="bilinear", antialias=True)
        x = F.relu(self.deconv6(x))
        x = self.dropout6(x)
        
        x = F.interpolate(x, size=self.last_upsample, mode="bilinear", antialias=True)
        x = F.relu(self.deconv7(x))
        x = self.deconv8(x)
        
        return x
  
 
class Autoencoder(torch.nn.Module):
    def __init__(self, out_channels: int, padding: bool, img_size: tuple, *args, **kwargs) -> None:
        """Autoencoder

        Args:
            out_channels (int): Number of output channels
            padding (bool): wether to use padding or not
            img_size (tuple): size of input images
        """
        super().__init__(*args, **kwargs)
    
        self.enc = encoder()
        self.dec = decoder(out_channels, padding, img_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent_space = self.enc(x)
        reconstruction = self.dec(latent_space)
        return reconstruction


if __name__ == '__main__':
    enc = encoder()
    # summary(enc, (3, 256, 256), device="cpu")
    
    dec = decoder(out_channels=3, padding=1, img_size=(256, 256))
    
    # summary(dec, (64, 1, 1), device="cpu")
    
    ae = Autoencoder(384, 1, (256, 256)) 
    
    summary(ae, (3, 256, 256), device="cpu")