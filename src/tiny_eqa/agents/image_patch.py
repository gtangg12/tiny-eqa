from __future__ import annotations

import numpy as np

from tiny_eqa.agents.common import *
from tiny_eqa.agents.image_patch_functions import *


class ImagePatch:
    """ Represents a crop of an image centered around a particular object, and relevant information, analogous to ScenePatch.
    """
    def __init__(self, image: np.ndarray, point1: Point2D = None, point2: Point2D = None):
        # Implementation specific to ImagePatch
        self.image = image
        self.height, self.width = self.image.shape[:2]
        self.point1 = point1 or np.zeros(2)
        self.point2 = point2 or (self.width, self.height)
        self.center = (point1 + point2) / 2

    def crop(self, point1: Point2D, point2: Point2D) -> ImagePatch:
        # Implementation specific to ImagePatch
        return ImagePatch(self.image, point1, point2)
    
    def distance(self, patch: ImagePatch) -> float:
        # Implementation specific to ImagePatch
        return np.linalg.norm(self.center - patch.center)

    def find(self, object_name: str) -> list[ImagePatch]:
        # Implementation specific to ImagePatch
        return image_find(self, object_name)

    def exists(self, object_name: str) -> bool:
        # Implementation specific to ImagePatch
        return len(self.find(object_name)) > 0
    
    def check_condition(self, object_name: str, condition: str) -> bool:
        # Implementation specific to ImagePatch
        return image_check_condition(self, object_name, condition)

    def text_match(self, text: str) -> float:
        # Implementation specific to ImagePatch
        return image_text_match(self, text)

    def simple_qa(self, question: str = None) -> str:
        # Implementation specific to ImagePatch
        return image_simple_qa(self, question)

    def depth(self) -> float:
        """ Returns the median depth of the object in the image.

        Examples:
        >>> # Which is farther, the cookie or the muffin?
        >>> def execute_command(image) -> float:
        >>>     image_patch = ImagePatch(image)
        >>>     cookie = image_patch.find('cookie')[0]
        >>>     muffin = image_patch.find('muffin')[0]
        >>>     return 'cookie' if cookie.depth() > muffin.depth() else 'muffin'
        """
        return image_compute_depth(self).median()