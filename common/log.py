import logging
from pathlib import Path


def create_logger(
    name: str, level=logging.INFO, filename: Path = None
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter(
        "[%(asctime)s] %(name)s (%(levelname)s): %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if filename is not None:
        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
