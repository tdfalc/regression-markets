from pathlib import Path

import numpy as np
import matplotlib as mpl


def save_figure(
    fig, savedir: Path, filename: str, dpi: int = 600, tight: bool = True
):
    if tight:
        fig.tight_layout()
    for extension in (".pdf", ".png"):
        fig.savefig(savedir / (filename + extension), dpi=dpi)


def add_dummy(X: np.ndarray):
    return np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)


def set_plot_style():
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(
        color=[
            "#F8766D",
            "#9590FF",
            "#A3A500",
            "#D89000",
            "#39B600",
            "#00BF7D",
            "#00BFC4",
            "#00B0F6",
            "#E76BF3",
        ]
    )
