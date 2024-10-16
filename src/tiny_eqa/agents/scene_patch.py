from __future__ import annotations

import numpy as np

from src.tiny_eqa.data import Scene
from tiny_eqa.agents.common import *
from tiny_eqa.agents.image_patch import ImagePatch
from tiny_eqa.agents.scene_patch_functions import *


class ScenePatch:
    """ Represents a crop of a scene centered around a particular object and relevant information.
    """
    def __init__(self, scene: Scene, point1: Point3D = None, point2: Point3D = None):
        """ Initializes a ScenePatch representing a crop of the scene, defined by the top-left and bottom-right corners. 
        If no points are provided, the entire scene is used.
        
        Parameters:
            scene: Scene
                The scene to crop.
            point1: Point3D
                The top left corner of the scene crop.
            point2: Point3D
                The bottom right corner of the crop.
        """
        self.scene = scene
        self.height, self.width, self.depth = self.scene.shape
        self.point1 = point1 or np.zeros(3)
        self.point2 = point2 or (self.width, self.height, self.depth)
        self.center = (point1 + point2) / 2

    def crop(self, point1: Point3D, point2: Point3D) -> ScenePatch:
        """ Returns a new ScenePatch cropped from the current ScenePatch.
        
        Parameters:
            point1: Point3D
                The top left corner of the new crop.
            point2: Point3D
                The bottom right corner of the new crop.
        """
        return ScenePatch(self.scene, self.point1, self.point2)
    
    def distance(self, patch: ScenePatch) -> float:
        """ Returns the distance between the centers of two ScenePatches.
        
        Parameters:
            patch: ScenePatch
                The ScenePatch to calculate the distance to.

        Examples:
        >>> # Calculate the distance between two doors
        >>> def execute_command(scene) -> float:
        >>>     scene_patch = ScenePatch(scene)
        >>>     doors = scene_patch.find('door')
        >>>     return doors[0].distance(doors[1])
        """
        return np.linalg.norm(self.center - patch.center)
    
    def find(self, object_name: str) -> list[ScenePatch]:
        """ Returns a list of ScenePatch objects matching object_name contained in the current ScenePatch if any are found.
        Otherwise, returns an empty list.
        
        Parameters:
            object_name: str
                The name of the object to find.
        
        Examples:
        >>> # return the cabinets
        >>> def execute_command(scene) -> List[ScenePatch]: 
        >>>     scene_patch = ScenePatch(scene)
        >>>     cabinets = scene_patch.find('cabinets')
        >>>     return cabinets
        """
        return scene_find(self, object_name)

    def exists(self, object_name: str) -> bool:
        """ Returns True if the object specified by object_name is found in the scene, and False otherwise.
        
        Parameters:
            object_name: str
                The name of the object to find.
        
        Examples:
        >>> # Are there both cabinets and chairs in the scene
        >>> def execute_command(scene) -> str:
        >>>     scene_patch = ScenePatch(scene)
        >>>     condition1 = scene_patch.exists('cabinets')
        >>>     condition2 = scene_patch.exists('chairs')
        >>>     return bool_to_yesno(condition1 and condition2)
        """
        return len(self.find(object_name)) > 0
    
    def check_condition(self, object_name: str, condition: str) -> bool:
        """ Returns True if the object possesses the property, and False otherwise.
        Differs from ’exists’ in that it presupposes the existence of the object specified by object_name, instead checking whether the object
        possesses the property.

        Parameters:
            object_name: str
                The name of the object to check.
            condition: str
                The property to check for.

        Examples:
        >>> # Are all the cabinets closed
        >>> def execute_command(scene) -> str:
        >>>     scene_patch = ScenePatch(scene)
        >>>     cabinets = scene_patch.find('cabinets')
        >>>     for cabinet in cabinets:
        >>>         if not cabinet.check_condition('cabinet', 'closed'):
        >>>             return 'no'
        >>>     return 'yes'
        """
        return scene_check_condition(self, object_name, condition)

    def text_match(self, text: str) -> float:
        """ Returns a similarity score in range [0, 1] between the text and the scene.
        Differs from check_condition in that it is useful for ranking matches between a text and multiple scenes and vice versa.

        Parameters:
            text: str
                The text to compare against the scene.
        
        Examples:
        >>> # Find the clothing article that is best described as something you would wear in the winter
        >>> def execute_command(scene) -> ScenePatch:
        >>>     scene_patch = ScenePatch(scene)
        >>>     text = 'Something you would wear in the winter.'
        >>>     clothes = scene_patch.find('clothes')
        >>>     match_scores = [clothes.text_match(text) for clothes in clothes]
        >>>     return clothes[np.argmax(match_scores)]
        """
        return scene_text_match(self, text)

    def simple_qa(self, question: str = None) -> str:
        """ Returns the answer to a basic question asked about the scene. If no question is provided, returns the answer to 'What is this?'.

        Parameters:
            question: str
                The question to ask about the scene.
        
        Examples:
        >>> # Which is the color of the book on the table
        >>> def execute_command(scene) -> str:
        >>>     scene_patch = ScenePatch(scene)
        >>>     tables = scene_patch.find('table')
        >>>     for table in tables:
        >>>         if table.exists('book'):
        >>>             return table.simple_qa('What color is the book?')
        >>>     return 'No book found.'
        """
        return scene_simple_qa(self, question)


def render(camera_position: Point3D, target_position: Point3D) -> ImagePatch:
    """ Returns the rendered image of the scene from the specified camera position looking at the target position.

    Parameters:
        camera_position: Point3D
            The position of the camera.
        target_position: Point3D
            The position of the target the camera is looking at.

    Examples:
        >>> # Standing next to the door, and looking towards the kitchen stove, is there anything cooking?
        >>> def execute_command(scene) -> str:
        >>>     scene_patch = ScenePatch(scene)
        >>>     door = scene_patch.find('door')[0]
        >>>     kitchen_stove = scene_patch.find('kitchen stove')[0]
        >>>     view = render(door.center, kitchen_stove.center)
        >>>     return view.simple_qa('Is anything cooking?')
    """
    return scene_render(camera_position, target_position)