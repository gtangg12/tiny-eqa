import numpy as np

from tiny_eqa.agents.common import *
from tiny_eqa.agents.image_patch_functions import ImagePatch


class ScenePatch: # typing w/o circular imports
    pass


def function_find(scene: ScenePatch, object_name: str) -> list[ScenePatch]:
    """
    """
    pass


def function_check_condition(scene: ScenePatch, object_name: str, condition: str) -> bool:
    """
    """
    pass


def function_text_match(scene: ScenePatch, text: str) -> float:
    """
    """
    pass


def function_simple_qa(scene: ScenePatch, question: str = None) -> str:
    """
    """
    pass


def function_render(scene: ScenePatch, camera_position: Point3D) -> ImagePatch:
    """
    """
    pass