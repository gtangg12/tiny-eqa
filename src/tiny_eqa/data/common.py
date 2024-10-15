from pathlib import Path

from omegaconf import OmegaConf
from torchtyping import TensorType


NumpyTensor = TensorType
TorchTensor = TensorType


def parent(path: Path, n: int = 1) -> Path:
    """ 
    Returns the n-th parent of the given path.
    """
    for _ in range(n):
        path = path.parent
    return path


CONFIGS_DIR = parent(Path(__file__), 4) / 'configs'


def load_config(path: Path | str, configT: type):
    """
    """
    config = OmegaConf.load(path)
    return OmegaConf.merge(OmegaConf.structured(configT), config)