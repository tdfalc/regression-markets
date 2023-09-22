from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

from config import get_config

DPI = get_config().PLOTS.dpi


def _save_fig(fig, save_location: Path = None, **kwargs):
    if save_location is not None:
        fig.savefig(save_location, bbox_inches="tight", dpi=DPI, **kwargs)
    plt.close(fig)


def _plot_series(y: np.ndarray, color: str, xlabel: str, ylabel: str, title: str):
    fig, ax = plt.subplots()
    ax.plot(y, c=color)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    return fig


def plot_target(y: np.ndarray, title: str, save_location: Path = None):
    fig = _plot_series(y, "red", "Timestep", "y", title)
    _save_fig(fig, save_location)


def plot_payments_evolution(payments: np.ndarray, title: str, save_location: Path = None):
    fig = _plot_series(payments, None, "Timestep", "Payments (EUR)", title)
    _save_fig(fig, save_location)


def plot_weights_evolution(weights: np.ndarray, title: str, save_location: Path = None):
    fig, ax = plt.subplots()
    lines = ax.plot(weights if weights.ndim > 1 else weights.reshape(1, -1))
    ax.legend(iter(lines), [f"$\\omega_{i}$" for i in range(len(lines))])
    ax.set(xlabel="Timestep", ylabel="$\\omega_i$", title=title)
    _save_fig(fig, save_location)


def plot_payments_final(payments: np.ndarray, title: str, save_location: Path = None):
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(payments)), payments)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set(xlabel="Payment ID", ylabel="EUR", title=title)
    ax.grid(ls="--", c="k", alpha=0.5)
    _save_fig(fig, save_location)
