import torch
from torch.utils.data import Dataset, ConcatDataset
from .DatasetFromImage import DatasetFromImage
import torchvision

import os

from helpers.terminalColor import terminalColor as tc


def getAllPNGFiles(root: str) -> list[str]:
    """Lists all PNG Files inside a given ``root`` directory.

    Args:
        root (str): Directory to images

    Returns:
        list[str]: list of all PNG files inside directory
    """
    AllFiles = os.listdir(root)
    for file in AllFiles[:]:
        if not(file.endswith(".png")):
            AllFiles.remove(file)
    return AllFiles

def constructApollo15Dataset(root: str="/home/tsa/data/LRO/NAC/Apollo 15/", imgSize: int=256, stride: int=128, transform=None) -> Dataset:
    """Creates an Dataset from all ``.png`` images inside the given root. 

    Args:
        root (str, optional): Path to images. Defaults to "/home/tsa/data/LRO/NAC/Apollo 15/".
        imgSize (int, optional): resulting image size from large scale images. Defaults to 256.
        stride (int, optional): stride to go through large scale images. Defaults to 128.
        transform (optional): transformation, that should be applied before return. Defaults to None.

    Returns:
        Dataset: Dataset from NAC images
    """
    pngFiles = getAllPNGFiles(root)
    
    win_size = 15
    
    print(tc.info + "Loading .png files ...")
    
    allDatasets: list[DatasetFromImage] = []
    
    for file in pngFiles:
        currentImg = torchvision.io.read_image(os.path.join(root, file)).to(torch.float32) / 255.0
        currentImg = currentImg[0, None, ...]

        currentName = file.split(".")[0]
        currentDataset = DatasetFromImage(currentImg, imgSize=imgSize, stride=stride, transform=transform, name=currentName)
        allDatasets.append(currentDataset)

    print(tc.info + "Creating dataset ...")
    data = allDatasets[0]
    for cData in allDatasets[1:]:
        data += cData

    return data
            
def constructApollo15DatasetWOApollo(root: str, imgSize: int=256, stride: int=128, transform=None) -> Dataset:
    Name1 = "M175252641_cut1"
    Img1 = torchvision.io.read_image(os.path.join(root, Name1 + ".png")).to(torch.float32) / 255.0
    data1 = DatasetFromImage(Img1[0, None, :, 200:-650], imgSize=imgSize, stride=stride, transform=transform, name=Name1)
    
    Name2 = "M175252641_cut2"
    Img2 = torchvision.io.read_image(os.path.join(root, Name2 + ".png")).to(torch.float32) / 255.0
    data2 = DatasetFromImage(Img2[0, None, :, 360:-470], imgSize=imgSize, stride=stride, transform=transform, name=Name1)

    Name3 = "M175252641_cut3"
    Img3 = torchvision.io.read_image(os.path.join(root, Name3 + ".png")).to(torch.float32) / 255.0
    data3 = DatasetFromImage(Img3[0, None, :, 520:-330], imgSize=imgSize, stride=stride, transform=transform, name=Name1)

    # This one is the Apollo Landing site image
    Name4 = "M175252641_cut4"
    Img4 = torchvision.io.read_image(os.path.join(root, Name4 + ".png")).to(torch.float32) / 255.0
    data4 = DatasetFromImage(Img4[0, None, 4970:, 700:7276], imgSize=imgSize, stride=stride, transform=transform, name=Name1)

    Name5 = "M175252641_cut5"
    Img5 = torchvision.io.read_image(os.path.join(root, Name5 + ".png")).to(torch.float32) / 255.0
    data5 = DatasetFromImage(Img5[0, None, 0:8404, 907:7306], imgSize=imgSize, stride=stride, transform=transform, name=Name1)
    
    return ConcatDataset([data1, data2, data3, data4, data5])

def constructApollo15LandingSite(root: str, imgSize: int=256, stride: int=128, transform=None) -> DatasetFromImage:
    """Wrapper for loading Apollo 15 Landing site

    Args:
        root (str): Root to NAC files
        imgSize (int, optional): Image size, that should be used. Defaults to 256.
        stride (int, optional): stride for dataset generation. Defaults to 128.
        transform (optional): Transform, that should be applied before returning. Defaults to None.

    Returns:
        DatasetFromImage: Dataset object
    """
    # Create Dataset
    selectedImage = torchvision.io.read_image(os.path.join(root, "M175252641_cut4.png"))[0, None, ...].to(torch.float) / 255.0
    # Limit only to Apollo 15 Area
    selectedImage = selectedImage[:, 3457:4273, 3230:4046] 
    data = DatasetFromImage(selectedImage, imgSize=imgSize, stride=stride, transform=transform, name="Apollo landing site")
    
    return data

def constructLargeApollo15LandingSite(root: str, imgSize: int=256, stride: int=128, transform=None) -> DatasetFromImage:
    """Wrapper for loading Apollo 15 Landing site (a larger cut out)

    Args:
        root (str): Root to NAC files
        imgSize (int, optional): Image size, that should be used. Defaults to 256.
        stride (int, optional): stride for dataset generation. Defaults to 128.
        transform (optional): Transform, that should be applied before returning. Defaults to None.

    Returns:
        DatasetFromImage: Dataset object
    """
    # Create Dataset
    selectedImage = torchvision.io.read_image(os.path.join(root, "M175252641_cut4.png"))[0, None, ...].to(torch.float) / 255.0
    # Limit only to Apollo 15 Area
    selectedImage = selectedImage[:, 2809:4777, 1858:5170]
    data = DatasetFromImage(selectedImage, imgSize=imgSize, stride=stride, transform=transform, name="Apollo landing site")
    
    return data


if __name__ == '__main__':
    data = constructApollo15Dataset()
    print(tc.success + 'finished!')