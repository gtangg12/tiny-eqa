import numpy as np
from omegaconf import OmegaConf

from tiny_eqa.agents.common import *
from src.tiny_eqa.data.sequence import *
from tiny_eqa.agents.image_patch_functions import ImagePatch


class ScenePatch: # typing w/o circular imports
    pass


def scene_find(scene: ScenePatch, object_name: str) -> list[ScenePatch]:
    """
    look thru all images, if found object in image, (can do bbox projection as precheck) get depth, deproject to 3D and compute which 3D label it corresponds to

    take all 3D labels and compute center/bbox, check if each of those centers are within the scene bbox
    """
    pass


def scene_check_condition(scene: ScenePatch, object_name: str, condition: str) -> bool:
    """
    first call scene find, next for each of those labels in scene patch, check if condition is true (look at 2d crops derived from labels)
    
    database caches find crops for scene patches
    """
    pass


def scene_text_match(scene: ScenePatch, text: str) -> float:
    """
    if entire scene/find crops use that else use bbox of projected bbox, mean of clip scores (fast, batch no cache)
    
    for batching dont forget rescale crop
    """
    pass


def scene_simple_qa(scene: ScenePatch, question: str = None) -> str:
    """
    if entire scene/find crops use that else bbox of projected bbox, vlm decoding

    extract object names for hyperedges (contains reasoning plus question, use sim search to get cache)

    for batching dont forget rescale crops
    """
    pass


def scene_render(camera_position: Point3D, target_position: Point3D) -> ImagePatch:
    """
    TODO actually doesn't render, but returns closest image in the posed multiview sequence
    """
    pass