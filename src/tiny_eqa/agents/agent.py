from dataclasses import dataclass, field
from omegaconf import OmegaConf

from tiny_eqa.data.sequence import FrameSequence
from tiny_eqa.models.model_gpt import ModelGptInput, ModelGpt, unpack_content


class Program:
    """
    """
    def __init__(self, code: str):
        """
        """
        self.function = eval(code) # TODO parse

    def execute_command(self, sequence: FrameSequence):
        """
        """
        return function(sequence)


class Agent:
    """
    """
    def __init__(self, config: OmegaConf):
        """
        """
        self.config = config

    def __call__(self, task: str, sequence: FrameSequence):
        """
        """
        program = self.process(task)
        return program.execute_command(sequence)

    def process(self, task: str) -> Program:
        """
        """
        pass

    def reset(self):
        """
        """
        pass


if __name__ == '__main__':
    pass