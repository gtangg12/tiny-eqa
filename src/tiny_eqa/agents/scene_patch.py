from __future__ import annotations

import numpy as np

from tiny_eqa.agents.common import *
from tiny_eqa.agents.image_patch import ImagePatch
from tiny_eqa.agents.scene_patch_functions import *


class ScenePatch:
    """
    """
    def __init__(self, scene: Scene, point1: Point3D = None, point2: Point3D = None):
        """
        """
        self.scene = scene
        self.height, self.width, self.depth = self.scene.shape
        self.point1 = point1 or (0, 0, 0)
        self.point2 = point2 or (self.width, self.height, self.depth)
        self.center = Point3D(
            (self.point1.x + self.point2.x) / 2, 
            (self.point1.y + self.point2.y) / 2,
            (self.point1.z + self.point2.z) / 2,
        )

    def crop(self, point1: Point3D, point2: Point3D) -> ScenePatch:
        """
        """
        return ScenePatch(self.scene, self.point1, self.point2)
    
    def distance(self, patch: ScenePatch) -> float:
        """
        """
        return (
            self.center.x - patch.center.x,
            self.center.y - patch.center.y,
            self.center.z - patch.center.z,
        )
    
    def find(self, object_name: str) -> list[ScenePatch]:
        """
        """
        return function_find(self, object_name)

    def exists(self, object_name: str) -> bool:
        """
        """
        return len(self.find(object_name)) > 0
    
    def check_condition(self, object_name: str, condition: str) -> bool:
        """
        """
        return function_check_condition(self, object_name, condition)

    def text_match(self, text: str) -> float:
        """
        """
        return function_text_match(self, text)

    def simple_qa(self, question: str = None) -> str:
        """
        """
        return function_simple_qa(self, question)

    def render(self, camera_position: Point3D) -> ImagePatch:
        """
        """
        return function_render(self, camera_position)