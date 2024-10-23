import torch
import torchvision


class DarkenTransform(object):
    """Darkens the area of a path. 
    """
    def __init__(self, num_points: int=6, thickness: int=2) -> None:
        """Randomly creates a mask with a stipe on it. Will then darken the area of the mask (stripe) by 50%. 

        Args:
            num_points (int, optional): Number of random points to generate for path. Defaults to 6.
            thickness (int, optional): Thickness of drawn path. Defaults to 2.
        """
        self.num_points = num_points
        self.thickness = thickness
    
    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        n, c, h, w = batch.shape
        transformed_batch = batch.clone()
        
        randomEdgePoints = self.__createRandomEdgePoints(w=w, h=h)
        
        for i in range(0, n):
            random_idx = torch.randint(0, 4, (1,))
            last_point = randomEdgePoints[random_idx, :][0]
            points = torch.randint(low=0, high=w, size=(self.num_points, 2), dtype=torch.float)
            mask = torch.zeros(h, w)

            for j in range(0, self.num_points):
                x, y = points[j, :]
                mask = self._wu_line(mask, last_point[1], last_point[0], x, y, self.thickness)
                last_point = torch.tensor([y, x])
            idx = torch.where(mask==1)
            transformed_batch[i, :, idx[0], idx[1]] = transformed_batch[i, :, idx[0], idx[1]] * 0.5
        
        return transformed_batch
    
    def __createRandomEdgePoints(self, w: int, h: int) -> torch.Tensor:
        possibleLastPoints = torch.zeros(4, 2)
        
        # Left edge
        possibleLastPoints[0, 1] = torch.randint(0, h-self.thickness, (1,))
        # Right edge 
        possibleLastPoints[1, 0] = w - self.thickness
        possibleLastPoints[1, 1] = torch.randint(0, h-self.thickness, (1,))
        # Top edge
        possibleLastPoints[2, 0] = torch.randint(0, w-self.thickness, (1,))
        # Bottom edge 
        possibleLastPoints[3, 1] = h - self.thickness
        possibleLastPoints[3, 0] = torch.randint(0, w-self.thickness, (1,))
        
        return possibleLastPoints
    
    def _wu_line(self, mask: torch.Tensor, x0: int, y0: int, x1: int, y1: int, thickness: int=1) -> torch.Tensor:
        """
        Draws a thick line from point (x0, y0) to point (x1, y1) using Wu's line algorithm.
        
        Args:
            mask (torch.Tensor): The image array to draw the line on.
            x0 (int): The x-coordinate of the starting point.
            y0 (int): The y-coordinate of the starting point.
            x1 (int): The x-coordinate of the ending point.
            y1 (int): The y-coordinate of the ending point.
            thickness (int, optional): The thickness of the line. Defaults to 1.
            
        Returns:
            torch.Tensor: The modified image array with the line drawn.
        """
        mask = mask.clone()
        dx = x1 - x0
        dy = y1 - y0
        length = max(abs(dx), abs(dy))
        
        if length == 0:
            return mask
        
        dx /= length
        dy /= length
        
        x, y = x0, y0
        
        for _ in range(int(length) + 1):
            for j in range(-thickness // 2, thickness // 2 + 1):
                for i in range(-thickness // 2, thickness // 2 + 1):
                    if 0 <= x + i < mask.shape[1] and 0 <= y + j < mask.shape[0]:
                        mask[int(y + j), int(x + i)] = 1
            x += dx
            y += dy
        
        return mask
        

class DarkenBlobTransform(object):
    """Darkens a complete circle randomly placed in the image. 
    """
    def __init__(self, width: tuple[int, int]=(5, 20)) -> None:
        self.min_r, self.max_r = width
    
    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        n, c, h, w = batch.shape
        
        idxX = torch.randint(0, w-self.min_r, (n,))
        idxY = torch.randint(0, h-self.min_r, (n,))
        darkening_factor = torch.randint(15, 45, (n,)) / 100.0 
        
        x = torch.arange(0, w)
        y = torch.arange(0, h)
        
        transformed_batch = batch.clone()
        for i in range(0, n):
            r = torch.randint(self.min_r, self.max_r, (1,))
            
            mask = (x[None,:]-idxX[i])**2 + (y[:, None]-idxY[i])**2 < r**2
            transformed_batch[i, :, mask] = transformed_batch[i, :, mask] * darkening_factor[i]

        
        return transformed_batch


class WhitenBlobTransform(object):
    """Whitens a complete circle randomly placed in the image. 
    """
    def __init__(self, width: tuple[int, int]=(3, 8)) -> None:
        self.min_r, self.max_r = width
    
    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        n, c, h, w = batch.shape
        
        idxX = torch.randint(0, w-self.min_r, (n,))
        idxY = torch.randint(0, h-self.min_r, (n,))

        x = torch.arange(0, w)
        y = torch.arange(0, h)
        
        transformed_batch = batch.clone()
        for i in range(0, n):
            r = torch.randint(self.min_r, self.max_r, (1,))
            
            mask = (x[None,:]-idxX[i])**2 + (y[:, None]-idxY[i])**2 < r**2
            transformed_batch[i, :, mask] = 0.8

        
        return transformed_batch
    

if __name__ == "__main__":
    transform = DarkenTransform()
    
    grayscale = torchvision.io.read_image("transforms/testMoon.png")[0, ...] / 255.0
    grayscale = grayscale[None, None, ...]
    # grayscale = torch.rand((1, 1, 224, 224))
    
    transformed_img = transform(grayscale)
    
    import matplotlib.pyplot as plt
    plt.subplot(1, 2, 1)
    plt.imshow(grayscale[0, 0])
    plt.title("Original image")
    plt.subplot(1, 2, 2)
    plt.imshow(transformed_img[0, 0])
    plt.title("Transformed image")
    plt.savefig("test.png")

