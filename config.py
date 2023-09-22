from functools import lru_cache
from pathlib import Path

from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig


@lru_cache(maxsize=1)
def _load_config() -> DictConfig:
    config_file = Path(__file__).parent / "config.yml"
    config = OmegaConf.load(config_file)
    return config  # type: ignore


def get_config() -> DictConfig:
    return _load_config().copy()
