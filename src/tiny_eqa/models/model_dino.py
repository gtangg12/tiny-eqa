from dataclasses import dataclass
from typing import Literal

import torch
from PIL import Image

from tiny_eqa.data.common import TorchTensor
from tiny_eqa.models.transforms import transform_imagenet
from tiny_eqa.utils.math import min_max_norm, pca_transform


def visualize_dino(features: TorchTensor['Hout', 'Wout', 'emb']) -> Image.Image:
    """
    Given a batch of dino features, visualizes features' first 3 (RGB) principal components.
    """
    H, W, dim = features.shape
    features = torch.from_numpy(features.detach().cpu().numpy())
    features = features.reshape(-1, dim)
    
    pca_features = pca_transform(features, n=3)
    pca_features = min_max_norm(pca_features, dim=-1)
    pca_features = pca_features.reshape(H, W, 3)
    return Image.fromarray((pca_features * 255).astype('uint8'))


@dataclass
class DinoModelConfig:
    """ Dino feature channels dimension. """
    FEATURE_CHANNELS = 1024

    """ Dino patch size. The output features have dims (Hout, Wout) / PATCH_SIZE. """
    PATCH_SIZE = 14

    """ Dino vision transformer backbone. """
    backbone: Literal['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14'] = 'dinov2_vitl14'

    """ Rescale images to (H, W) // `downsample` resolution, which in turns controls feature resolution. 
        The training set consists of 224x244 images (16x16 features)-- the closer the images/feature resolution to 
        these, the higher the feature quality. """
    downsample: float = 2

    def input_dims(self, image: TorchTensor['batch', 'ch', 'H', 'W']) -> tuple[int, int]:
        """ 
        Returns the input image dimensions that are compatible with dino.
        """
        round_dim = lambda x: x // self.PATCH_SIZE * self.PATCH_SIZE
        _, _, H, W = image.shape
        return (
            round_dim(int(H / self.downsample)),
            round_dim(int(W / self.downsample)),
        )


class DinoModel:
    """
    """
    def __init__(self, config: DinoModelConfig, device='cuda'):
        """
        """
        self.config = config
        self.device = device
        self.model = torch.hub.load('facebookresearch/dinov2', self.config.backbone)
        self.model.eval()
        self.model.to(device)

    def __call__(self, image: TorchTensor['batch', 'ch', 'H', 'W']) -> TorchTensor['batch', 'Hout', 'Wout', 'dim']:
        """
        """
        image = self.transform(image).to(self.device)
        
        features: TorchTensor['batch', 'Hout x Wout', 'dim'] = self.model.forward_features(image)['x_norm_patchtokens']
        features = features / features.norm(dim=-1, keepdim=True)
        features = features.reshape(
            -1, 
            image.shape[2] // self.config.PATCH_SIZE, 
            image.shape[3] // self.config.PATCH_SIZE, 
            self.config.FEATURE_CHANNELS,
        )
        return features

    def transform(self, image: TorchTensor['batch', 'ch', 'H', 'W']):
        """
        Transforms an image to be compatible with dino vision transformer. 
        """
        Hout, Wout = self.config.input_dims(image)
        transform = transform_imagenet(resize=(Hout, Wout))
        return transform(image)
    

if __name__ == '__main__':
    from torchvision import transforms
    image = Image.open('tests/scene.png').convert('RGB')
    model = DinoModel(DinoModelConfig())
    batch = transforms.ToTensor()(image).unsqueeze(0)
    outputs = model(batch)
    visualize_dino(outputs[0]).save('tests/scene_dino.png')