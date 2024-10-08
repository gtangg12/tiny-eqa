import numpy as np
from omegaconf import OmegaConf


class ImagePatch: # typing w/o circular imports
    """
    """
    config = OmegaConf.load('configs/image_patch.yaml')


def image_find(image: ImagePatch, object_name: str) -> list[ImagePatch]:
    """
    """
    pass


def image_check_condition(image: ImagePatch, object_name: str, condition: str) -> bool:
    """
    """
    pass


def image_text_match(image: ImagePatch, text: str) -> float:
    """
    """
    pass


def image_simple_qa(image: ImagePatch, question: str = None) -> str:
    """
    """
    pass


def image_compute_depth(image: ImagePatch) -> np.ndarray:
    """
    """
    pass