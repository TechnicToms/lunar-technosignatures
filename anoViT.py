import torch
import torchvision
from torchmetrics import StructuralSimilarityIndexMeasure

import math
import os

from lightning import LightningModule
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from cutAndPaste import CutAndPasteDataLoader

from models.decoder_anoViT import AnoViTDecoder


class AnoViT(LightningModule):
    def __init__(self, hparams: dict):
        super(AnoViT, self).__init__()
        for key in hparams.keys():
            self.hparams[key]=hparams[key]
            
        self.vit = torchvision.models.VisionTransformer(image_size=self.hparams["imgSize"], patch_size=self.hparams["patch_size"], num_layers=self.hparams["num_layers"],
                                                        num_heads=self.hparams["num_heads"], hidden_dim=self.hparams["hidden_dim"], mlp_dim=self.hparams["embed_dim"])
        self.decoder = AnoViTDecoder(self.hparams["embed_dim"])
        
        self.loss_function = torch.nn.MSELoss()
        
        self.ssim_metric_train = StructuralSimilarityIndexMeasure()
        self.ssim_metric_test = StructuralSimilarityIndexMeasure()
        self.ssim_metric_valid = StructuralSimilarityIndexMeasure()
        
    def forward_encoder(self, x: torch.Tensor) -> torch.Tensor:
        x = self.vit._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.vit.encoder(x)
        
        return x
        
    def forward_decoder(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)
        
    def forward(self, x: torch.Tensor):
        # Forward encoder
        x = self.forward_encoder(x)
        
        # Reshape
        n, n_seq, c = x.shape
        side_length = int(math.sqrt(n_seq - 1))
        encoded_embeddings = x[:, 1:n_seq, :].reshape((n, side_length, side_length, c))
        encoded_embeddings = encoded_embeddings.permute(0, 3, 1, 2)     # New shape (n, c, side_length, side_length)
        
        # Forward decoder
        reconstructed_image = self.forward_decoder(encoded_embeddings)
        
        return reconstructed_image
    
    def on_fit_start(self):
        self.logger.log_hyperparams(self.hparams)

    def training_step(self, batch, batch_idx):
        # Feed forward
        stacked_batch = torch.stack([batch[:, 0, ...]]*3, dim=1)
        reconstruction = self(stacked_batch)
        
        # Compute loss
        loss = self.loss_function(reconstruction, stacked_batch)
        
        self.ssim_metric_train(reconstruction, stacked_batch)
        self.log("train_ssim", self.ssim_metric_train, logger=True, on_epoch=True, on_step=True)
        self.log("train_loss", loss, logger=True, on_epoch=True, on_step=True)
        
        tensorboard = self.logger.experiment
        if batch_idx % 350 == 0:
            tensorboard.add_image('reconstructed_image', reconstruction[0, ...], global_step=self.global_step, dataformats="CHW")
        
        if self.global_step % 500 == 0:
            self.trainer.save_checkpoint(filepath=os.path.join(self.logger.log_dir, "checkpoints/AnoViT.ckpt"))
        
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        # Feed forward
        stacked_batch = torch.stack([batch[:, 0, ...]]*3, dim=1)
        reconstruction = self(stacked_batch)
        
        # Compute loss
        loss = self.loss_function(reconstruction, stacked_batch)
        
        self.ssim_metric_valid(reconstruction, stacked_batch)
        self.log("validation_ssim", self.ssim_metric_train, logger=True)
        self.log("validation_loss", loss, logger=True)
        
        return {'val_loss': loss}

    def test_step(self, batch, batch_idx):
        stacked_batch = torch.stack([batch[:, 0, ...]]*3, dim=1)
        # Feed forward
        reconstruction = self(stacked_batch)
        
        # Compute loss
        loss = self.loss_function(reconstruction, stacked_batch)
        
        self.ssim_metric_test(reconstruction, stacked_batch)
        self.log("test_ssim", self.ssim_metric_train, logger=True)
        self.log("test_loss", loss, logger=True)
        
        return {'test_loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)



if __name__ == '__main__':
    hparams = {
        # Data related
        'root': '/home/tsa/data/LRO/NAC/Apollo 17/',
        'imgSize': 224,
        'stride': 28,
        
        # Transformer related
        'patch_size': 16, 
        'num_layers': 4,
        'num_heads': 4,
        'hidden_dim': 128,
        'embed_dim': 128,
        
        # Training related
        'lr': 1e-4,
        'batch_size': 64,
        'num_epochs': 75,
        
        # Logger
        'loggerTag': 'AnoViT',
    }

    model = AnoViT(hparams=hparams)

    loader = CutAndPasteDataLoader(hparams["root"], imgSize=hparams["imgSize"], stride=hparams["stride"], batch_size=hparams["batch_size"])  

    # latest_checkpoint = ModelCheckpoint(filename="latest-{epoch}-{step}", monitor="step", every_n_train_steps=50, save_top_k=-1)
    logger = TensorBoardLogger(save_dir='tb_logs', name=hparams['loggerTag'], version="AnoViT-Apollo17")
    tr = Trainer(max_epochs=hparams["num_epochs"], logger=logger, enable_checkpointing=True)    # callbacks=[latest_checkpoint]
    tr.fit(model=model, datamodule=loader)
    
    print("finished!")