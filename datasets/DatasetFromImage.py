import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader

from defecty.core import terminalColor as tc


class DatasetFromImage(Dataset):
    def __init__(self, img: torch.Tensor, imgSize: int=512, stride:int=512, transform=None, name: str ="img") -> None:
        """Splits up an a large scale image (``img``) into patches of ``imgSize x imgSize`` and creates a dataset from it.

        Args:
            img (torch.Tensor): Input image from which the Dataset should be created.
            imgSize (int, optional): Image size. Defaults to 512.
            stride (int, optional): stride to go though original image. Defaults to 512.
            transform (_type_, optional): Transformation, that will be applied to image before returning. Defaults to None.
            name (str, optional): Name of dataset. Defaults to "img".
        """
        self.imgSize = imgSize
        self.stride = stride
        self.transform = transform
        
        self.name = name
        
        self.num_cols = None
        self.num_rows = None
        
        self.img = img
        self.patchIds = self.__generatePatchIds()
    
    def __repr__(self):
        return tc.info + f"DatasetFromImage('{self.name}', #{self.__len__()}, stride={self.stride})"
    
    def __len__(self):
        return self.patchIds.shape[1]
    
    def __getitem__(self, index) -> torch.Tensor:
        if isinstance(index, torch.Tensor):
            index = index.tolist()
            
        if isinstance(index, int):
            patches = self.__extract_patches_from_indices([index])
            if self.transform:
                patches = self.transform(patches)
            return patches[0, ...]
        elif isinstance(index, list):
            patches = self.__extract_patches_from_indices(index)
            if self.transform:
                patches = self.transform(patches)
            return patches 
        elif isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            fullIndex = [i for i in range(start, stop, step)]
            patches = self.__extract_patches_from_indices([fullIndex])
            if self.transform:
                patches = self.transform(patches)
            return patches  
        else:
            raise TypeError(tc.err + f"Input type of index ({type(index)}) is none of the following: torch.Tensor, int, list or slice") 
    
    def __add__(self, other: Dataset) -> ConcatDataset:
        return ConcatDataset([self, other])
    
    def __generatePatchIds(self) -> torch.Tensor:
        c, h, w = self.img.shape
        
        xArr = torch.arange(0, w-self.imgSize, self.stride)
        yArr = torch.arange(0, h-self.imgSize, self.stride)
        
        self.num_cols = len(xArr)
        self.num_rows = len(yArr)
        
        xMesh, yMesh = torch.meshgrid(xArr, yArr, indexing="ij")
        
        points = torch.stack([xMesh.flatten(), yMesh.flatten()])
        return points
    
    def __extract_patches_from_indices(self, idx: list) -> torch.Tensor: 
        c, h, w = self.img.shape
        allPatches = torch.zeros(len(idx), c, self.imgSize, self.imgSize)
        for k, i in enumerate(idx):
            cx, cy = self.patchIds[:, i]
            current_patch = self.img[:, cy:cy+self.imgSize, cx:cx+self.imgSize]
            allPatches[k, ...] = current_patch
        
        return allPatches


if __name__ == '__main__':
    img = torch.rand(3, 1024, 1024)
    data = DatasetFromImage(img, imgSize=512, stride=256)
    print(data[1].shape)
    print(tc.success + 'finished!')