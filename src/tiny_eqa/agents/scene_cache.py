from dataclasses import dataclass, field


class ImagePatch: # typing w/o circular imports
    pass


class ScenePatch: # typing w/o circular imports
    pass


class EmbeddingDatabase:
    """
    """
    pass


@dataclass
class ScenePatchData:
    """
    """

    """ """
    name: str = None
    """ """
    views: list[ImagePatch] = None
    """ """
    conditions: EmbeddingDatabase = field(default_factory=EmbeddingDatabase)
    """ """
    questions : EmbeddingDatabase = field(default_factory=EmbeddingDatabase)


@ dataclass
class SceneCache:
    """
    """
    name2patch: dict[str, ScenePatch]
    patch2data: dict[ScenePatch, ScenePatchData]