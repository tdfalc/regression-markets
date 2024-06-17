from pathlib import Path
from collections import defaultdict
from enum import Enum
from typing import Callable, Any, Sequence

import numpy as np
from matplotlib.colors import to_hex
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt

from market.data import BatchData


def set_style() -> None:
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    plt.rc("font", size=12)
    plt.rc("axes", labelsize=12)
    plt.rc("xtick", labelsize=12)
    plt.rc("ytick", labelsize=12)
    plt.rc("legend", fontsize=12)


class MarketDesigns(str, Enum):
    mle_nll = r"$\mathcal{M}^{\mathrm{MLE}}_{\mathrm{NLL}}$"
    blr_nll = r"$\mathcal{M}^{\mathrm{BLR}}_{\mathrm{NLL}}$"
    blr_kld_m = r"$\mathcal{M}^{\mathrm{BLR}}_{\mathrm{KL}-m}$"
    blr_kld_c = r"$\mathcal{M}^{\mathrm{BLR}}_{\mathrm{KL}-v}$"


def save_figure(
    fig, savedir: Path, filename: str, dpi: int = 300, tight: bool = True
) -> None:
    if tight:
        fig.tight_layout()
    for extension in (".pdf", ".png"):
        fig.savefig(savedir / (filename + extension), dpi=dpi, transparent=True)


def add_dummy(X: np.ndarray[float]) -> np.ndarray[float]:
    return np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)


def get_discrete_colors(cmap_name: str, num_colors: int) -> Sequence:
    cmap = get_cmap(cmap_name)
    return [to_hex(cmap(i)) for i in range(0, cmap.N, cmap.N // num_colors)]


def nested_defaultdict(levels: int, default_factory: Any) -> defaultdict:
    if levels == 1:
        return defaultdict(default_factory)
    return defaultdict(lambda: nested_defaultdict(levels - 1, default_factory))


def build_input(sample_size: int, num_features: int) -> np.ndarray[float]:
    mean, covariance = np.zeros(num_features), np.eye(num_features)
    X = np.random.multivariate_normal(mean, covariance, size=sample_size)
    return add_dummy(X)


def build_data(
    train_size: int,
    test_size: float,
    num_features: int,
    interpolant_function: Callable,
    additive_noise_function: Callable,
    heteroskedasticity_function: Callable,
) -> Callable:
    sample_size = train_size + test_size
    test_frac = test_size / sample_size

    def _build_data():
        X = build_input(sample_size, num_features)
        noise = additive_noise_function(sample_size)
        y = interpolant_function(X)
        y += noise * heteroskedasticity_function(X)
        return BatchData(X[:, [0]], X[:, [1]], X[:, 2:], y, test_frac=test_frac)

    return _build_data


def conditional_value_at_risk(x: np.ndarray, alpha: float, axis: int = 0) -> np.ndarray:
    return -np.mean(
        np.partition(x, int(x.shape[axis] * alpha), axis=axis)[
            : int(x.shape[axis] * alpha)
        ],
        axis=axis,
    )


def bootstrap_resample(
    arr: np.ndarray, num_bootstraps: int, sample_size: int = None
) -> np.ndarray:
    if sample_size is None:
        sample_size = len(arr)
    idx = np.random.randint(sample_size, size=(num_bootstraps, sample_size)).T
    return arr[idx]
