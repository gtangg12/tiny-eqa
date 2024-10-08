from abc import ABC, abstractmethod
from pathlib import Path

from tiny_eqa.agents import Agent
from tiny_eqa.data.sequence import FrameSequence


class Benchmark(ABC):
    """
    """
    def __init__(self, path: Path | str):
        """
        """
        pass

    @abstractmethod
    def __call__(self, agent: Agent) -> dict:
        """
        """
        pass

    @abstractmethod
    def run(self, task: str, sequence: FrameSequence) -> dict:
        """
        """
        pass