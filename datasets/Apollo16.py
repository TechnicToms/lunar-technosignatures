import torch
from torch.utils.data import Dataset, ConcatDataset
from .DatasetFromImage import DatasetFromImage
import torchvision

import os

from helpers.terminalColor import terminalColor as tc


def constructApollo16DatasetWOApollo(root: str, imgSize: int=256, stride: int=128, transform=None) -> Dataset:
    """Creates a Dataset from the NAC images without the Apollo 16 Landing site.

    Args:
        root (str): Root to NAC files
        imgSize (int, optional): Image size, that should be used. Defaults to 256.
        stride (int, optional): stride for dataset generation. Defaults to 128.
        transform (optional): Transform, that should be applied before returning. Defaults to None.

    Returns:
        Dataset: Dataset object
    """
    Name1 = "M175179080_cut1"
    Img1 = torchvision.io.read_image(os.path.join(root, Name1 + ".png")).to(torch.float32) / 255.0
    data1 = DatasetFromImage(Img1[0, None, 131:, 157:5798], imgSize=imgSize, stride=stride, transform=transform, name=Name1)
    
    # This one contains the apollo 16 landing site
    Name2 = "M175179080_cut2"
    Img2 = torchvision.io.read_image(os.path.join(root, Name2 + ".png")).to(torch.float32) / 255.0
    data2 = DatasetFromImage(Img2[0, None, 0:12491, 245:6079], imgSize=imgSize, stride=stride, transform=transform, name=Name1)

    Name3 = "M175179080_cut3"
    Img3 = torchvision.io.read_image(os.path.join(root, Name3 + ".png")).to(torch.float32) / 255.0
    data3 = DatasetFromImage(Img3[0, None, :, 325:6067], imgSize=imgSize, stride=stride, transform=transform, name=Name1)

    Name4 = "M175179080_cut4"
    Img4 = torchvision.io.read_image(os.path.join(root, Name4 + ".png")).to(torch.float32) / 255.0
    data4 = DatasetFromImage(Img4[0, None, :, 305:6101], imgSize=imgSize, stride=stride, transform=transform, name=Name1)

    Name5 = "M175179080_cut5"
    Img5 = torchvision.io.read_image(os.path.join(root, Name5 + ".png")).to(torch.float32) / 255.0
    data5 = DatasetFromImage(Img5[0, None, 0:14558, 240:6096], imgSize=imgSize, stride=stride, transform=transform, name=Name1)
    
    return ConcatDataset([data1, data2, data3, data4, data5])

def constructApollo16LandingSite(root: str, imgSize: int=256, stride: int=128, transform=None) -> DatasetFromImage:
    """Wrapper for loading Apollo 16 Landing site

    Args:
        root (str): Root to NAC files
        imgSize (int, optional): Image size, that should be used. Defaults to 256.
        stride (int, optional): stride for dataset generation. Defaults to 128.
        transform (optional): Transform, that should be applied before returning. Defaults to None.

    Returns:
        DatasetFromImage: Dataset object
    """
    # Create Dataset
    selectedImage = torchvision.io.read_image(os.path.join(root, "M175179080_cut2.png"))[0, None, ...].to(torch.float) / 255.0
    # Limit only to Apollo 16 Area
    selectedImage = selectedImage[:, 12379:13195, 3337:4153]
    data = DatasetFromImage(selectedImage, imgSize=imgSize, stride=stride, transform=transform, name="Apollo 16 landing site")
    
    return data

def constructLargeApollo16LandingSite(root: str, imgSize: int=256, stride: int=128, transform=None) -> DatasetFromImage:
    """Wrapper for loading Apollo 16 Landing site (a larger cut out)

    Args:
        root (str): Root to NAC files
        imgSize (int, optional): Image size, that should be used. Defaults to 256.
        stride (int, optional): stride for dataset generation. Defaults to 128.
        transform (optional): Transform, that should be applied before returning. Defaults to None.

    Returns:
        DatasetFromImage: Dataset object
    """
    # Create Dataset
    selectedImage = torchvision.io.read_image(os.path.join(root, "M175179080_cut2.png"))[0, None, ...].to(torch.float) / 255.0
    # Limit only to Apollo 16 Area
    selectedImage = selectedImage[:, 12342:13238, 2848:4640]
    data = DatasetFromImage(selectedImage, imgSize=imgSize, stride=stride, transform=transform, name="Apollo 16 landing site")
    
    return data


if __name__ == '__main__':
    print(tc.success + 'finished!')