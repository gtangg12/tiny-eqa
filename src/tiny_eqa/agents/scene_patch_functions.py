import numpy as np
from omegaconf import OmegaConf

from tiny_eqa.agents.common import *
from src.tiny_eqa.data.sequence import *
from tiny_eqa.agents.image_patch_functions import ImagePatch


class ScenePatch: # typing w/o circular imports
    """
    """
    config = OmegaConf.load('configs/scene_patch.yaml')


def scene_find(scene: ScenePatch, object_name: str) -> list[ScenePatch]:
    """
    """
    pass


def scene_check_condition(scene: ScenePatch, object_name: str, condition: str) -> bool:
    """
    """
    pass


def scene_text_match(scene: ScenePatch, text: str) -> float:
    """
    """
    pass


def scene_simple_qa(scene: ScenePatch, question: str = None) -> str:
    """
    """
    pass


def scene_render(camera_position: Point3D, target_position: Point3D) -> ImagePatch:
    """
    TODO actually doesn't render, but returns closest image in the posed multiview sequence
    """
    pass