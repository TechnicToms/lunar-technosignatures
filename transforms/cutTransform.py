import torch
import torchvision


class CutPatchTransform(object):
    """Cuts out a patch of given image.
    """
    def __init__(self, patch_size: int=64) -> None:
        """Cuts out a single patch of size ``(patch_size, patch_size)`` at a random position in the image, and replace the values with 0.5.

        Args:
            patch_size (int, optional): Used patch size. Defaults to 64.
        """
        self.patchSize = patch_size
    
    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        n, c, h, w = batch.shape
        idxX = torch.randint(0, w-self.patchSize, (n,))
        idxY = torch.randint(0, h-self.patchSize, (n,))
        
        transformed_batch = batch.clone()
        for i in range(0, n):
            transformed_batch[i, :, idxY[i]:idxY[i]+self.patchSize, idxX[i]:idxX[i]+self.patchSize] = 0.5
        
        return transformed_batch


class CutPatchMeanTransform(object):
    """Replaces patch area with mean.
    """
    def __init__(self, patch_size: int=64) -> None:
        """Cuts out a single patch of size ``(patch_size, patch_size)`` at a random position in the image, 
        and replace the values with the mean of the corresponding area.

        Args:
            patch_size (int, optional): Used patch size. Defaults to 64.
        """
        self.patchSize = patch_size
    
    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        n, c, h, w = batch.shape
        idxX = torch.randint(0, w-self.patchSize, (n,))
        idxY = torch.randint(0, h-self.patchSize, (n,))
        
        transformed_batch = batch.clone()
        for i in range(0, n):
            transformed_batch[i, :, idxY[i]:idxY[i]+self.patchSize, idxX[i]:idxX[i]+self.patchSize] = torch.mean(batch[i, :, idxY[i]:idxY[i]+self.patchSize, idxX[i]:idxX[i]+self.patchSize])
        
        return transformed_batch
    
    
if __name__ == "__main__":
    transform = CutPatchMeanTransform(patch_size=64)
    
    grayscale = torch.rand((1, 1, 224, 224))
    
    transformed_img = transform(grayscale)
    
    import matplotlib.pyplot as plt
    plt.subplot(1, 2, 1)
    plt.imshow(grayscale[0, 0])
    plt.title("Original image")
    plt.subplot(1, 2, 2)
    plt.imshow(transformed_img[0, 0])
    plt.title("Transformed image")
    plt.savefig("test.png")
    


