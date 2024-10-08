from omegaconf import OmegaConf


class Model:
    """
    """
    def __init__(self, config: OmegaConf, device=None):
        """
        """
        self.config = config
        self.device = device

    def __call__(self):
        """
        """
        pass