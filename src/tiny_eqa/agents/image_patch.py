from __future__ import annotations
from collections import namedtuple

import numpy as np

from tiny_eqa.agents.common import *
from tiny_eqa.agents.image_patch_functions import *


class ImagePatch:
    """
    """
    def __init__(self, image: np.ndarray, point1: Point2D = None, point2: Point2D = None):
        """
        """
        self.image = image
        self.height, self.width = self.image.shape[:2]
        self.point1 = point1 or (0, 0)
        self.point2 = point2 or (self.width, self.height)
        self.center = Point2D(
            (self.point1.x + self.point2.x) / 2, 
            (self.point1.y + self.point2.y) / 2,
        )

    def crop(self, point1: Point2D, point2: Point2D) -> ImagePatch:
        """
        """
        return ImagePatch(self.image, point1, point2)
    
    def distance(self, patch: ImagePatch) -> tuple[float, float]:
        """
        """
        return (
            self.center.x - patch.center.x,
            self.center.y - patch.center.y,
        )

    def find(self, object_name: str) -> list[ImagePatch]:
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

    def object_depth(self) -> float:
        """
        """
        return function_compute_depth(self).median()