from collections import namedtuple

from src.tiny_eqa.agents.common_functions import *


Point2D = namedtuple('Point2D', ['x', 'y'])
Point3D = namedtuple('Point3D', ['x', 'y', 'z'])


def bool_to_yesno(expression: bool) -> str:
    return 'yes' if expression else 'no'


def content_qa(question: str) -> str:
    """
    """
    return function_content_qa(question)