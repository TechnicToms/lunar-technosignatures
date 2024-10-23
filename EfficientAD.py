import torch
import torchvision

from torch.utils.data import DataLoader, Dataset, Subset
from torchdata.datapipes.iter import Zipper, IterableWrapper
from torch.utils.data import random_split
from datasets.Apollo17 import constructApollo17DatasetWOApollo

from lightning import LightningModule, LightningDataModule
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger

from models.PDN import PDNSmall, PDNMedium, EfficientAdModelSize
from models.autoencoders import Autoencoder
from datasets.ImageFolder import InfiniteDataloader, ImageFolderWithoutTarget

import os
import random

from helpers.terminalColor import terminalColor as tc


class EffAD(LightningModule):
    def __init__(self, hparams: dict):
        super(EffAD, self).__init__()
        for key in hparams.keys():
            self.hparams[key]=hparams[key]
        
        self.teacher: PDNSmall | PDNMedium
        self.student: PDNSmall | PDNMedium
        self.pad_maps = True
        self.input_size = (hparams["imgSize"], hparams["imgSize"])

        if hparams["model_size"] == EfficientAdModelSize.S:
            self.teacher = PDNSmall(out_channels=hparams["teacher_channels"], padding=hparams["padding"]).eval()
            self.student = PDNSmall(out_channels=2 * hparams["teacher_channels"], padding=hparams["padding"])
        elif hparams["model_size"] == EfficientAdModelSize.M:
            self.teacher = PDNMedium(out_channels=hparams["teacher_channels"], padding=hparams["padding"]).eval()
            self.student = PDNMedium(out_channels=2 * hparams["teacher_channels"], padding=hparams["padding"])
        else:
            raise ValueError(f"modelSize can be 'S' or 'M' but got {hparams['model_size']}!")
        
        state_dict = torch.load(hparams['teacher_ckpt'], map_location='cpu', weights_only=True)
        self.teacher.load_state_dict(state_dict)
        
        self.ae: Autoencoder = Autoencoder(out_channels=hparams["teacher_channels"], padding=hparams["padding"], img_size=self.input_size)
        self.teacher_channels: int = hparams["teacher_channels"]
        
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
    
    def reduce_tensor_elems(self, tensor: torch.Tensor, m=2**24) -> torch.Tensor:
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
    
    def forward_train(self, batch: torch.Tensor, imageNetBatch: torch.Tensor, distance_st: torch.Tensor) -> dict[str, torch.Tensor]:
        # Student loss
        distance_st = self.reduce_tensor_elems(distance_st)
        d_hard = torch.quantile(distance_st, 0.999)
        loss_hard = torch.mean(distance_st[distance_st >= d_hard])
        student_output_penalty = self.student(imageNetBatch)[:, : self.teacher_channels, :, :]
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
    
    def forward_predict(self, batch: torch.Tensor, distance_st: torch.Tensor, student_output: torch.Tensor) -> dict[str, torch.Tensor]:
        """Computes anomaly score and anomaly maps.

        Args:
            batch (torch.Tensor): Input batch in (N, C, H, W) format.
            distance_St (torch.Tensor): Distances of student. 
            student_output (torch.Tensor): Output of student.

        Returns:
            dict[str, torch.Tensor]: Dictionary with the following keys:
                - anomaly_map: Global and local anomaly map combined.
                - map_st: Anomaly map student (local anomaly map).
                - map_ae: Anomaly map autoencoder (global anomaly map).
        """
        with torch.no_grad():
            ae_output = self.ae(batch)

        map_st = torch.mean(distance_st, dim=1, keepdim=True)
        map_stae = torch.mean(
            (ae_output - student_output[:, self.teacher_channels :]) ** 2, dim=1, keepdim=True
        )

        if self.pad_maps:
            map_st = torch.nn.functional.pad(map_st, (4, 4, 4, 4))
            map_stae = torch.nn.functional.pad(map_stae, (4, 4, 4, 4))
        map_st = torch.nn.functional.interpolate(map_st, size=(self.input_size[0], self.input_size[1]), mode="bilinear", antialias=True)
        map_stae = torch.nn.functional.interpolate(map_stae, size=(self.input_size[0], self.input_size[1]), mode="bilinear", antialias=True)

        if self.is_set(self.quantiles):
            map_st = 0.1 * (map_st - self.quantiles["qa_st"]) / (self.quantiles["qb_st"] - self.quantiles["qa_st"])
            map_stae = (
                0.1 * (map_stae - self.quantiles["qa_ae"]) / (self.quantiles["qb_ae"] - self.quantiles["qa_ae"])
            )

        map_combined = 0.75 * map_st + 0.25 * map_stae
        return {"anomaly_map": map_combined, "map_st": map_st, "map_ae": map_stae}
    
    def forward(self, batch: torch.Tensor, imageNetBatch: torch.Tensor=None) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            teacher_output = self.teacher(batch)
            if self.is_set(self.mean_std):
                teacher_output = (teacher_output - self.mean_std["mean"]) / self.mean_std["std"]

        student_output = self.student(batch)
        distance_st = torch.pow(teacher_output - student_output[:, : self.teacher_channels, :, :], 2)
        
        return self.forward_train(batch=batch, imageNetBatch=imageNetBatch, distance_st=distance_st)    

    def on_fit_start(self):
        self.logger.log_hyperparams(self.hparams)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx):
        lunar_batch, imageNet_batch = batch
        lunar_batch = torch.stack([lunar_batch[:, 0, ...]]*3, dim=1)
        
        out_dict: dict[str, torch.Tensor] = self(lunar_batch, imageNet_batch)
        
        loss = out_dict["loss_student"] + out_dict["loss_autoencoder"] + out_dict["loss_student_autoencoder"]
        
        self.log("train_loss_student", out_dict["loss_student"], logger=True, on_step=True)
        self.log("train_loss_autoencoder", out_dict["loss_autoencoder"], logger=True, on_step=True)
        self.log("train_loss_student_autoencoder", out_dict["loss_student_autoencoder"], logger=True, on_step=True)
        self.log("train_loss", loss, logger=True, on_step=True)
        
        if self.global_step % 500 == 0:
            self.trainer.save_checkpoint(filepath=os.path.join(self.logger.log_dir, "checkpoints/effAD.ckpt"))
        
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        lunar_batch, imageNet_batch = batch
        lunar_batch = torch.stack([lunar_batch[:, 0, ...]]*3, dim=1)
        
        out_dict: dict[str, torch.Tensor] = self(lunar_batch, imageNet_batch)
        
        loss = out_dict["loss_student"] + out_dict["loss_autoencoder"] + out_dict["loss_student_autoencoder"]
        
        self.log("val_loss_student", out_dict["loss_student"], logger=True, on_step=True)
        self.log("val_loss_autoencoder", out_dict["loss_autoencoder"], logger=True, on_step=True)
        self.log("val_loss_student_autoencoder", out_dict["loss_student_autoencoder"], logger=True, on_step=True)
        self.log("val_loss", loss, logger=True, on_step=True)
        
        return {'val_loss': loss}

    def test_step(self, batch, batch_idx):
        lunar_batch, imageNet_batch = batch
        lunar_batch = torch.stack([lunar_batch[:, 0, ...]]*3, dim=1)
        
        out_dict: dict[str, torch.Tensor] = self(lunar_batch, imageNet_batch)
        
        loss = out_dict["loss_student"] + out_dict["loss_autoencoder"] + out_dict["loss_student_autoencoder"]
        
        self.log("test_loss_student", out_dict["loss_student"], logger=True, on_step=True)
        self.log("test_loss_autoencoder", out_dict["loss_autoencoder"], logger=True, on_step=True)
        self.log("test_loss_student_autoencoder", out_dict["loss_student_autoencoder"], logger=True, on_step=True)
        self.log("test_loss", loss, logger=True, on_step=True)
        
        return {'test_loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def predict_step(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Computes anomaly score and anomaly maps.

        Args:
            x (torch.Tensor): Input batch in (N, C, H, W) format.

        Returns:
            dict[str, torch.Tensor]: Dictionary with the following keys:
                - anomaly_map: Global and local anomaly map combined.
                - map_st: Anomaly map student (local anomaly map).
                - map_ae: Anomaly map autoencoder (global anomaly map).
        """
        x = torch.stack([x[:, 0, ...]]*3, dim=1)
        with torch.no_grad():
            teacher_output = self.teacher(x)
            if self.is_set(self.mean_std):
                teacher_output = (teacher_output - self.mean_std["mean"]) / self.mean_std["std"]

        student_output = self.student(x)
        distance_st = torch.pow(teacher_output - student_output[:, : self.teacher_channels, :, :], 2)
        
        return self.forward_predict(batch=x, distance_st=distance_st, student_output=student_output)


class ApolloDataLoader(LightningDataModule):
    def __init__(self, root: str, root_imageNet: str, imgSize: int, stride: int, batch_size=32, num_workers=5):
        super().__init__()
        self.root = root
        self.root_imageNet = root_imageNet
        self.imgSize = imgSize
        self.stride = stride
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def setup(self, stage=None):
        transform_lunar = torchvision.transforms.Normalize(mean=[0.2987], std=[0.0922])
        dataset = constructApollo17DatasetWOApollo(self.root, self.imgSize, self.stride, transform=transform_lunar)
        print(tc.info + f"Got {len(dataset)} of images in total.")
        self.train, self.valid, self.test = random_split(dataset, (0.8, 0.1, 0.1))
        
        penalty_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((2 * self.imgSize, 2 * self.imgSize)),
            torchvision.transforms.RandomGrayscale(0.3),
            torchvision.transforms.CenterCrop(self.imgSize),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        dataImageNet = ImageFolderWithoutTarget(self.root_imageNet, transform=penalty_transform)
        indices = torch.multinomial(torch.ones(len(dataImageNet)), len(dataset), replacement=False)
        dataImageNetSubset = Subset(dataset=dataImageNet, indices=indices)
        self.imNetTrain, self.imNetVal, self.imNetTest = random_split(dataImageNetSubset, (0.8, 0.1, 0.1))

    def train_dataloader(self):
        # return zip(DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers), self.loaderPenalty)
        dataLunarTrain = IterableWrapper(self.train)
        dataImNetTrain = IterableWrapper(self.imNetTrain)
        return DataLoader(dataLunarTrain.zip(dataImNetTrain), batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        
    def val_dataloader(self):
        #return zip(DataLoader(self.valid, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers), self.loaderPenalty)
        dataLunarVal = IterableWrapper(self.valid)
        dataImNetVal = IterableWrapper(self.imNetVal)
        return DataLoader(dataLunarVal.zip(dataImNetVal), batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        # return zip(DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers), self.loaderPenalty)
        dataLunarTest = IterableWrapper(self.test)
        dataImNetTest = IterableWrapper(self.imNetTest)
        return DataLoader(dataLunarTest.zip(dataImNetTest), batch_size=self.batch_size, num_workers=self.num_workers)



if __name__ == '__main__':
    hparams = {
        # Data related
        'root': '/home/tsa/data/LRO/NAC/Apollo 17/',
        'root_imageNet': '/home/tsa/data/imagenet/ILSVRC/Data/CLS-LOC/train',
        'imgSize': 256,
        'stride': 28,
        
        # Model related stuff
        'teacher_channels': 384,
        'model_size': EfficientAdModelSize.S,
        'padding': True,
        'teacher_ckpt': 'models/teacher_checkpoints/teacher_small_final_state.pth',
        
        # Training related
        'lr': 1e-4,
        'batch_size': 64,
        'num_epochs': 75,
        
        # Logger
        'loggerTag': 'EffAD',
    }

    model = EffAD(hparams=hparams)

    loader = ApolloDataLoader(hparams["root"], root_imageNet=hparams["root_imageNet"], imgSize=hparams["imgSize"], stride=hparams["stride"], batch_size=hparams["batch_size"])  

    logger = TensorBoardLogger(save_dir='tb_logs', name=hparams['loggerTag'], version="Apollo17")
    tr = Trainer(max_epochs=hparams["num_epochs"], logger=logger, enable_checkpointing=True)
    tr.fit(model=model, datamodule=loader)
    
    print("finished!")