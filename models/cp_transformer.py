import torch
import torchvision


class ViT(torch.nn.Module):
    def __init__(self, img_size: int, patch_size: int) -> None:
        super().__init__()
        self.backbone = torchvision.models.vision_transformer.VisionTransformer(image_size=img_size, patch_size=patch_size, num_layers=12, num_heads=12, 
                                                                                hidden_dim=768, mlp_dim=512, num_classes=2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return x
    
    def extractFeatures(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts all features after the encoders

        Args:
            x (torch.Tensor): input image of shape `(N, C, H, W)`

        Returns:
            torch.Tensor: output encoder embeddings `(N, H//patchSize * W//patchSize, embeddingDim)`
        """
        x = self.backbone._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.backbone.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.backbone.encoder(x)
        return x


if __name__ == '__main__':
    model = ViT(img_size=224, patch_size=16)
    model.eval()
    
    test = model(torch.rand(1, 3, 224, 224))
    
    print('finished!')