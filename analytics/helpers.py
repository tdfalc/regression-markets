from pathlib import Path
from collections import defaultdict
from enum import Enum
from typing import Any

import numpy as np
import matplotlib.pyplot as plt


def set_style() -> None:
    fontsize = 14
    plt.rcParams.update(
        {
            "text.latex.preamble": r"\usepackage{amsfonts, amsmath}",
            "font.size": fontsize,
            "axes.titlesize": fontsize,
            "axes.labelsize": fontsize,
            "xtick.labelsize": fontsize,
            "ytick.labelsize": fontsize,
            "legend.fontsize": fontsize,
            "figure.titlesize": fontsize,
        }
    )


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
    fig.savefig(
        savedir / f"{filename}.pdf",
        dpi=dpi,
        transparent=True,
        bbox_inches="tight",
    )


def add_dummy(X: np.ndarray[float]) -> np.ndarray[float]:
    return np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)


def nested_defaultdict(levels: int, default_factory: Any) -> defaultdict:
    if levels == 1:
        return defaultdict(default_factory)
    return defaultdict(lambda: nested_defaultdict(levels - 1, default_factory))


def build_input(sample_size: int, num_features: int) -> np.ndarray[float]:
    mean, covariance = np.zeros(num_features), np.eye(num_features)
    X = np.random.multivariate_normal(mean, covariance, size=sample_size)
    return add_dummy(X)


def bootstrap_resample(
    arr: np.ndarray, num_bootstraps: int, sample_size: int = None
) -> np.ndarray:
    if sample_size is None:
        sample_size = len(arr)
    idx = np.random.randint(sample_size, size=(num_bootstraps, sample_size)).T
    return arr[idx]
