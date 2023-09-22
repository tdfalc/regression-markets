from pathlib import Path

import numpy as np

from cases.run import run_case
from config import get_config


def main(case: str, directory: Path, use_cache: bool = False):
    run_case(case, directory, use_cache=use_cache)


if __name__ == "__main__":
    np.random.seed(seed=42)
    config = get_config().GENERAL
    main(config.case, Path(config.directory), use_cache=False)
