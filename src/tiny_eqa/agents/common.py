from collections import namedtuple

from src.tiny_eqa.agents.common_functions import *


Point2D = np.ndarray[2]
Point3D = np.ndarray[3]


def bool_to_yesno(expression: bool) -> str:
    """ Evaluate a boolean expression and return 'yes' or 'no'.

    Examples:
    >>> bool_to_yesno(condition1 and condition2 and condition3)
    """
    return 'yes' if expression else 'no'


def content_qa(question: str) -> str:
    """ Ask a general question that is self contained i.e. does not require a scene or image to answer.

    Examples:
    >>> content_qa('What is the capital of France?')
    >>> content_qa('Is a metal chair or fluffy stool more comfortable for sitting?')
    """
    return function_content_qa(question)