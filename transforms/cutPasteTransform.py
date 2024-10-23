import torch
import torchvision


class CutPasteTransform(object):
    def __init__(self, patch_size: int=64, use_flipping: bool=False) -> None:
        """Selects a random area in each image of size ``(patch_size, patch_size)`` and replaces a different random chosen Area of it in the same image.
        Args:
            patch_size (int, optional): Used patch size. Defaults to 64.
            use_flipping (bool, optional): If the pasted should also be flipped at random. Defaults to False
        """
        self.patchSize = patch_size
        self.flipping = use_flipping
        
        self.flipTransform = torchvision.transforms.Compose([
            torchvision.transforms.RandomVerticalFlip(p=0.5),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            ])
    
    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        n, c, h, w = batch.shape
        
        idxXPaste = torch.randint(0, w-self.patchSize, (n,))
        idxYPaste = torch.randint(0, h-self.patchSize, (n,))
        
        idxXCopy = torch.randint(0, w-self.patchSize, (n,))
        idxYCopy = torch.randint(0, h-self.patchSize, (n,))
        
        transformed_batch = batch.clone()
        for i in range(0, n):
            cutout = batch[i, :, idxYCopy[i]:idxYCopy[i]+self.patchSize, idxXCopy[i]:idxXCopy[i]+self.patchSize]
            if self.flipping:
                cutout = self.flipTransform(cutout[None,...])[0, ...]
            transformed_batch[i, :, idxYPaste[i]:idxYPaste[i]+self.patchSize, idxXPaste[i]:idxXPaste[i]+self.patchSize] = cutout
        
        return transformed_batch
    

if __name__ == "__main__":
    transform = CutPasteTransform(patch_size=64, use_flipping=True)
    
    grayscale = torchvision.io.read_image("testMoon.png") / 255.0
    grayscale = grayscale[None, 0:1, ...]
    
    transformed_img = transform(grayscale)
    
    import matplotlib.pyplot as plt
    plt.subplot(1, 2, 1)
    plt.imshow(grayscale[0, 0])
    plt.title("Original image")
    plt.subplot(1, 2, 2)
    plt.imshow(transformed_img[0, 0])
    plt.title("Transformed image")
    plt.savefig("test.png")