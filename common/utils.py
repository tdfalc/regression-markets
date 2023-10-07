from typing import Sequence, List, Tuple
import joblib
from pathlib import Path
import contextlib
import itertools
import os
import hashlib
from functools import wraps

import numpy as np
import cloudpickle


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def chain_combinations(
    sequence: Sequence[int], min: int, max: int, replace: bool = False
) -> List[Tuple]:
    iterator = (
        itertools.combinations_with_replacement if replace else itertools.combinations
    )
    return list(
        itertools.chain.from_iterable(
            (iterator(sequence, i) for i in range(min, max + 1))
        )
    )


def safe_divide(numerator: np.ndarray, denominator: np.ndarray):
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.true_divide(numerator, denominator)
        result[result == np.inf] = 0
    return np.nan_to_num(result)


def cache(
    save_dir: Path,
    use_cache=True,
    extra_mem_cache=True,
    kwargs_to_add: List[str] = None,
):
    mem_cache = {}

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__

            # Serialize arguments and keyword arguments using cloudpickle
            pickle_str = cloudpickle.dumps((args, kwargs))
            input_hash = hashlib.sha256(pickle_str).hexdigest()[:8]

            # Create a string for extra keyword arguments
            kwarg_string = ""
            if kwargs_to_add is not None:
                kwarg_string = "".join([f"_{k}={kwargs[k]}" for k in kwargs_to_add])

            # Construct the cache path
            cache_path = save_dir / f"{func_name}{kwarg_string}_{input_hash}.pkl"

            if use_cache:
                if extra_mem_cache and cache_path in mem_cache:
                    return mem_cache[cache_path]

                if cache_path.exists():
                    with open(cache_path, "rb") as f:
                        result = cloudpickle.load(f)
                        mem_cache[cache_path] = result
                        return result

            result = func(*args, **kwargs)
            mem_cache[cache_path] = result

            # Create the directory if it doesn't exist
            if not save_dir.exists():
                os.makedirs(save_dir, exist_ok=True)

            # Save the result to the cache
            with open(cache_path, "wb") as f:
                cloudpickle.dump(result, f)

            return result

        return wrapper

    return decorator
