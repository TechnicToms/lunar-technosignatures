import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies.ddp import DDPStrategy

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split
import torchmetrics.classification
from datasets.Apollo15 import constructApollo15DatasetWOApollo, constructApollo15LandingSite
from datasets.Apollo17 import constructApollo17DatasetWOApollo, constructApollo17LandingSite

import torchvision
from transforms.cutPasteTransform import CutPasteTransform
from transforms.cutTransform import CutPatchTransform, CutPatchMeanTransform
from transforms.darkenTransform import DarkenTransform, DarkenBlobTransform, WhitenBlobTransform

import torchmetrics
from evaluateModel import evaluateLandingSite

from helpers.terminalColor import terminalColor as tc


class CutAndPaste(pl.LightningModule):
    def __init__(self, patch_size: int):
        super().__init__()
        self.backbone = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = torch.nn.Flatten()
        
        self.freeze_for_n_epochs = 5
        # self.backbone.requires_grad_(False)
        self.backbone_is_frozen = True

        self.head = torch.nn.Sequential(torch.nn.Linear(512, 256), torch.nn.ReLU(),
                                        torch.nn.Linear(256, 128), torch.nn.ReLU(),
                                        torch.nn.Linear(128, 64),  torch.nn.ReLU(),
                                        torch.nn.Linear(64,  32),  torch.nn.ReLU(),
                                        torch.nn.Linear(32,  2),   torch.nn.Softmax(dim=1))
        
        self.transformations = [CutPasteTransform(patch_size=patch_size, use_flipping=True),
                                CutPatchMeanTransform(patch_size=patch_size), WhitenBlobTransform(width=(3, 8)),
                                DarkenTransform(num_points=6, thickness=2), DarkenBlobTransform(width=(5, 18))]
        
        self.normalization = torchvision.transforms.Normalize(mean=[0.31945744]*3, std=[0.13307516]*3)
        
        self.train_accuracy = torchmetrics.classification.Accuracy(task="binary", num_classes=2)
        self.train_precision = torchmetrics.classification.Precision(task="binary", num_classes=2)
        self.train_recall = torchmetrics.classification.Recall(task="binary", num_classes=2)
        
        self.valid_accuracy = torchmetrics.classification.Accuracy(task="binary", num_classes=2)
        self.valid_precision = torchmetrics.classification.Precision(task="binary", num_classes=2)
        self.valid_recall = torchmetrics.classification.Recall(task="binary", num_classes=2)
        
        self.test_accuracy = torchmetrics.classification.Accuracy(task="binary", num_classes=2)
        self.test_precision = torchmetrics.classification.Precision(task="binary", num_classes=2)
        self.test_recall = torchmetrics.classification.Recall(task="binary", num_classes=2)
            
    def __applyCutAndPaste(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n, _, _, _ = batch.shape
        labels = torch.zeros(n, device=self.device, dtype=torch.long)
        
        idxHalfWay = n // 2
        randomChosenTransformationIdx = torch.randint(0, len(self.transformations), (1,))
        batch[idxHalfWay:, ...] = self.transformations[randomChosenTransformationIdx](batch[idxHalfWay:, ...])
        labels[idxHalfWay:, ...] = 1
        
        return torch.stack([batch[:, 0, ...]]*3, dim=1), torch.nn.functional.one_hot(labels, num_classes=2).to(torch.float)
    
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        batch = self.normalization(batch)
        x = self.backbone(batch)
        x = self.head(x)
        return x
  
    def training_step(self, batch: torch.Tensor, batch_idx):
        imgs, labels = self.__applyCutAndPaste(batch)
        
        predictions = self(imgs)
        loss = torch.nn.functional.cross_entropy(predictions, labels)
        
        self.train_accuracy(predictions, labels)
        self.train_precision(predictions, labels)
        self.train_recall(predictions, labels)

        return loss
    
    def on_train_epoch_end(self):
        # log epoch metric
        self.log('train_accuracy', self.train_accuracy)
        self.log('train_precision', self.train_precision)
        self.log('train_recall', self.train_recall)
        
        if not self.backbone_is_frozen:
            if self.freeze_for_n_epochs == self.current_epoch:
                self.backbone.requires_grad_(True)
                self.backbone_is_frozen = True
        
    def validation_step(self, batch, batch_idx):
        x, y = self.__applyCutAndPaste(batch)
        y_hat = self(x)
        val_loss = torch.nn.functional.cross_entropy(y_hat, y)
        
        self.valid_accuracy(y_hat, y)
        self.valid_precision(y_hat, y)
        self.valid_recall(y_hat, y)
        
        self.log('val_loss', val_loss, sync_dist=True)
        self.log('val_accuracy', self.valid_accuracy, sync_dist=True)
        self.log('val_precision', self.valid_precision, sync_dist=True)
        self.log('val_recall', self.valid_recall, sync_dist=True)
        return val_loss
    
    def test_step(self, batch, batch_idx):
        x, y = self.__applyCutAndPaste(batch)
        y_hat = self(x)
        test_loss = torch.nn.functional.cross_entropy(y_hat, y)
        self.test_accuracy(y_hat, y)
        self.test_precision(y_hat, y)
        self.test_recall(y_hat, y)
        
        self.log('test_loss', test_loss)
        self.log('test_accuracy', self.test_accuracy)
        self.log('test_precision', self.test_precision)
        self.log('test_recall', self.test_recall)
        return test_loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-4, weight_decay=1e-5)
        return optimizer
 

class CutAndPasteDataLoader(pl.LightningDataModule):
    def __init__(self, root: str, imgSize: int, stride: int, batch_size=32, num_workers=4):
        super().__init__()
        self.root = root
        self.imgSize = imgSize
        self.stride = stride
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def setup(self, stage=None):
        dataset = constructApollo15DatasetWOApollo(self.root, self.imgSize, self.stride)
        print(tc.info + f"Got {len(dataset)} of images in total.")
        self.train, self.valid, self.test = random_split(dataset, (0.8, 0.1, 0.1))

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


if __name__ == "__main__":
    data_root = "/home/tsa/data/LRO/NAC/Apollo 15/"
    im_size = 224
    stride = 14
    batch_size = 32
    cut_patch_size = 32

    loader = CutAndPasteDataLoader(root=data_root, imgSize=im_size, stride=stride, batch_size=batch_size, num_workers=12)
    
    CutPasteModel = CutAndPaste(patch_size=cut_patch_size)
    
    # logger = TensorBoardLogger("tb_logs", name="CutAndPaste", version="Apollo15")
    # tr = pl.Trainer(logger=logger, strategy=DDPStrategy(), enable_checkpointing=True, default_root_dir="checkpoints", max_epochs=75)
    # tr.fit(model=CutPasteModel, datamodule=loader)
    # tr.test(model=CutPasteModel, datamodule=loader)

    
    test_data = constructApollo15LandingSite(root=data_root, imgSize=im_size, stride=8)
    loaded_model = CutAndPaste.load_from_checkpoint("tb_logs/CutAndPaste/version_1/checkpoints/epoch=75-step=934952.ckpt", patch_size=cut_patch_size)
    evaluateLandingSite(model=loaded_model, dataset=test_data, data_root=data_root)
    
    print(tc.success + "finished!")
