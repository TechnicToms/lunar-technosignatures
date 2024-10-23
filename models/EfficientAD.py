import torch
import torch.nn.functional as F
import torchvision

import random

# Import needed networks
from .PDN import PDNSmall, PDNMedium, EfficientAdModelSize
from .autoencoders import Autoencoder


def reduce_tensor_elems(tensor: torch.Tensor, m=2**24) -> torch.Tensor:
    """Flattens n-dimensional tensors,  selects m elements from it
    and returns the selected elements as tensor. It is used to select
    at most 2**24 for torch.quantile operation, as it is the maximum
    supported number of elements.
    https://github.com/pytorch/pytorch/blob/b9f81a483a7879cd3709fd26bcec5f1ee33577e6/aten/src/ATen/native/Sorting.cpp#L291

    Args:
        tensor (torch.Tensor): input tensor from which elements are selected
        m (int): number of maximum tensor elements. Default: 2**24

    Returns:
            Tensor: reduced tensor
    """
    tensor = torch.flatten(tensor)
    if len(tensor) > m:
        # select a random subset with m elements.
        perm = torch.randperm(len(tensor), device=tensor.device)
        idx = perm[:m]
        tensor = tensor[idx]
    return tensor


class EfficientAdModule(torch.nn.Module):
    def __init__(self, teacher_channels: int, input_size: tuple[int, int], modelSize: EfficientAdModelSize=EfficientAdModelSize.S, 
                 padding: bool=False, pad_maps: bool=True,) -> None:
        super().__init__()
        
        self.teacher: PDNSmall | PDNMedium
        self.student: PDNSmall | PDNMedium
        self.pad_maps = pad_maps

        if modelSize == EfficientAdModelSize.S:
            self.teacher = PDNSmall(out_channels=teacher_channels, padding=padding).eval()
            self.student = PDNSmall(out_channels=2 * teacher_channels, padding=padding)
        elif modelSize == EfficientAdModelSize.M:
            self.teacher = PDNMedium(out_channels=teacher_channels, padding=padding).eval()
            self.student = PDNMedium(out_channels=2 * teacher_channels, padding=padding)
        else:
            raise ValueError(f"modelSize can be 'S' or 'M' but got {modelSize}!")
        
        self.ae: Autoencoder = Autoencoder(out_channels=teacher_channels, padding=padding, img_size=input_size)
        self.teacher_channels: int = teacher_channels
        self.inputSize = input_size
        
        self.mean_std: torch.nn.ParameterDict = torch.nn.ParameterDict({
            "mean": torch.zeros((1, self.teacher_channels, 1, 1)),
            "std": torch.zeros((1, self.teacher_channels, 1, 1)),
        })
        
        self.quantiles: torch.nn.ParameterDict = torch.nn.ParameterDict({
            "qa_st": torch.tensor(0.0),
            "qb_st": torch.tensor(0.0),
            "qa_ae": torch.tensor(0.0),
            "qb_ae": torch.tensor(0.0),
        })

    def choose_random_aug_image(self, image: torch.Tensor) -> torch.Tensor:
        """Chooses a random augmentation and applies it to the given input image.

        Args:
            image (torch.Tensor): input image.

        Returns:
            torch.Tensor: Augmented input image
        """
        transform_functions = [
            torchvision.transforms.functional.adjust_brightness,
            torchvision.transforms.functional.adjust_contrast,
            torchvision.transforms.functional.adjust_saturation,
        ]
        # Sample an augmentation coefficient λ from the uniform distribution U(0.8, 1.2)
        coefficient = random.uniform(0.8, 1.2)  # nosec: B311
        transform_function = random.choice(transform_functions)  # nosec: B311
        return transform_function(image, coefficient)

    def is_set(self, p_dic: torch.nn.ParameterDict) -> bool:
        """Checks if the parameter dict contains values other than zero. If so: returns true, because then the parameters has been changed.

        Args:
            p_dic (torch.nn.ParameterDict): Parameter Dict 

        Returns:
            bool: if is changed returns True
        """
        for _, value in p_dic.items():
            if value.sum() != 0:
                return True
        return False

    def forward(self, batch: torch.Tensor, batchImageNet: torch.Tensor=None) -> torch.Tensor | dict:
        with torch.no_grad():
            teacher_output = self.teacher(batch)
            if self.is_set(self.mean_std):
                teacher_output = (teacher_output - self.mean_std["mean"]) / self.mean_std["std"]

        student_output = self.student(batch)
        distance_st = torch.pow(teacher_output - student_output[:, : self.teacher_channels, :, :], 2)

        if self.training:
            # Student loss
            distance_st = reduce_tensor_elems(distance_st)
            d_hard = torch.quantile(distance_st, 0.999)
            loss_hard = torch.mean(distance_st[distance_st >= d_hard])
            student_output_penalty = self.student(batchImageNet)[:, : self.teacher_channels, :, :]
            loss_penalty = torch.mean(student_output_penalty**2)
            loss_st = loss_hard + loss_penalty
            
            # Autoencoder and Student AE Loss
            aug_img = self.choose_random_aug_image(batch)
            ae_output_aug = self.ae(aug_img)
            
            with torch.no_grad():
                teacher_output_aug = self.teacher(aug_img)
                if self.is_set(self.mean_std):
                    teacher_output_aug = (teacher_output_aug - self.mean_std["mean"]) / self.mean_std["std"]

            student_output_ae_aug = self.student(aug_img)[:, self.teacher_channels :, :, :]

            distance_ae = torch.pow(teacher_output_aug - ae_output_aug, 2)
            distance_stae = torch.pow(ae_output_aug - student_output_ae_aug, 2)

            loss_ae = torch.mean(distance_ae)
            loss_stae = torch.mean(distance_stae)
            return {'loss_student': loss_st, 'loss_autoencoder': loss_ae, 'loss_student_autoencoder': loss_stae}

        else:
            with torch.no_grad():
                ae_output = self.ae(batch)

            map_st = torch.mean(distance_st, dim=1, keepdim=True)
            map_stae = torch.mean(
                (ae_output - student_output[:, self.teacher_channels :]) ** 2, dim=1, keepdim=True
            )

            if self.pad_maps:
                map_st = F.pad(map_st, (4, 4, 4, 4))
                map_stae = F.pad(map_stae, (4, 4, 4, 4))
            map_st = F.interpolate(map_st, size=(self.inputSize[0], self.inputSize[1]), mode="bilinear", antialias=True)
            map_stae = F.interpolate(map_stae, size=(self.inputSize[0], self.inputSize[1]), mode="bilinear", antialias=True)

            if self.is_set(self.quantiles):
                map_st = 0.1 * (map_st - self.quantiles["qa_st"]) / (self.quantiles["qb_st"] - self.quantiles["qa_st"])
                map_stae = (
                    0.1 * (map_stae - self.quantiles["qa_ae"]) / (self.quantiles["qb_ae"] - self.quantiles["qa_ae"])
                )

            map_combined = 0.75 * map_st + 0.25 * map_stae
            return {"anomaly_map": map_combined, "map_st": map_st, "map_ae": map_stae}

