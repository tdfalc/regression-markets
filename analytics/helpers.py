from pathlib import Path
from collections import defaultdict
from enum import Enum
from typing import Callable, Any, Sequence

import numpy as np
from matplotlib.colors import to_hex
from matplotlib.pyplot import get_cmap
import matplotlib.pyplot as plt

from regression_markets.market.data import BatchData


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


import subprocess
from typing import Optional
import itertools

from matplotlib import pyplot as plt


plt.rcParams["font.size"] = 12
plt.rcParams["mathtext.fontset"] = "cm"  # Use CM for math font.
plt.rcParams["figure.autolayout"] = True  # Use tight layouts.


PALETTE_1 = ["magenta", "blue", "darkorange", "limegreen", "black"]
PALETTE_2 = [
    "#000000",  # Black
    "#F5793A",  # Orange
    "#4BA6FB",  # Modified blue (original was #85C0F9)
    "#A95AA1",  # Pink
    "#689948",  # Green: not optimal for colour-blind people
]


def get_colors(cycle: bool = True):
    """Get chosen color pallete with optional cycle."""
    if cycle:
        return itertools.cycle(PALETTE_1)
    return PALETTE_1


def use_tex() -> None:
    """Use TeX for rendering."""
    if subprocess.call("which latex", shell=True) == 0:
        plt.rcParams["text.usetex"] = True  # Use TeX for rendering.
    else:
        pass


def tweak(
    legend: Optional[bool] = None,
    legend_loc: Optional[str] = None,
    spines: Optional[bool] = True,
    ticks: Optional[bool] = True,
    ax: Optional[bool] = None,
) -> None:
    """Tweak a plot.

    Args:
        legend (bool, optional): Show a legend if any labels are set.
        legend_loc (str, optional): Position of the legend. Defaults to "upper right".
        spines (bool, optional): Hide top and right spine. Defaults to `True`.
        ticks (bool, optional): Hide top and right ticks. Defaults to `True`.
        ax (axis, optional): Axis to tune. Defaults to `plt.gca()`.
    """

    if legend_loc is None:
        legend_loc = "upper right"

    if ax is None:
        ax = plt.gca()

    if legend is None:
        legend = len(ax.get_legend_handles_labels()[0]) > 0

    if legend:
        leg = ax.legend(
            facecolor="#eeeeee",
            edgecolor="#ffffff",
            framealpha=0.85,
            loc=legend_loc,
            labelspacing=0.25,
        )
        leg.get_frame().set_linewidth(0)

    if spines:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_lw(1)
        ax.spines["left"].set_lw(1)

    if ticks:
        ax.xaxis.set_ticks_position("bottom")
        ax.xaxis.set_tick_params(width=1)
        ax.yaxis.set_ticks_position("left")
        ax.yaxis.set_tick_params(width=1)
