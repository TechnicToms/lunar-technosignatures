import torch
from torch.utils.data import Dataset


def computeNormalization(data: Dataset, num_points: int=10000) -> tuple[float, float]:
    """Computes the mean and std over a given sample size over the dataset.

    Args:
        data (Dataset): Dataset class
        num_points (int, optional): Number of data points, that should be used for computation. Defaults to 10000.

    Returns:
        tuple[float, float]: mean, std
    """
    test_img: torch.Tensor = data[0]
    c, h, w = test_img.shape
    
    bunchOfImages = torch.zeros(num_points, c, h, w)
    perm = torch.randperm(len(data))
    indexes = perm[:num_points]
    
    for it, idx in enumerate(indexes):
        bunchOfImages[it, ...] = data[idx]
    
    mean = float(torch.mean(bunchOfImages, dim=(0, 2, 3)))
    std = float(torch.std(bunchOfImages, dim=(0, 2, 3)))
    
    return mean, std