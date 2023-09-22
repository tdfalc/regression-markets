from functools import wraps
from pathlib import Path
import os
import logging
import sys

import cloudpickle


def file_cache(save_location: Path, use_cache=True):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if use_cache:
                if save_location.exists():
                    with open(save_location, "rb") as f:
                        return cloudpickle.load(f)
            result = func(*args, **kwargs)
            if not save_location.exists():
                os.makedirs(save_location.parent, exist_ok=True)
            with open(save_location, "wb") as f:
                cloudpickle.dump(result, f)
            return result

        return wrapper

    return decorator


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(stream_handler)
    formatter = logging.Formatter("%(asctime)s:%(name)s:%(message)s")
    stream_handler.setFormatter(formatter)
    return logger
