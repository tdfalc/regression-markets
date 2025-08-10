import numpy as np

from regression_markets.common.utils import chain_combinations


def add_polynomial_features(X: np.ndarray, degree: int) -> np.ndarray:
    """Add polynomials of given degree to an array of features"""
    combinations = chain_combinations(
        np.arange(X.shape[1]), 1, degree, replace=True
    )
    return np.hstack(
        [X[:, c].prod(axis=1).reshape(-1, 1) for c in combinations]
    )
