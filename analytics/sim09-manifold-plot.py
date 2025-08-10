import os
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
import pandas as pd
from tfds.plotting import prettify, use_tex

from regression_markets.common.log import create_logger
from analytics.helpers import save_figure, set_style

set_style()
use_tex()


def plot_manifold(savedir: Path) -> None:
    rhos = [0, 0.9]

    vmin, vmax = -3, 3
    resolution = 100
    num_levels = 10

    fig, ax = plt.subplots(figsize=(6.5, 6), dpi=300)

    for i, rho in enumerate(rhos):
        mean = np.zeros(2)
        covariance = [[1, rho], [rho, 1]]
        dist = stats.multivariate_normal(mean=mean, cov=covariance)

        X = np.linspace(vmin, vmax, resolution)
        Y = np.linspace(vmin, vmax, resolution)
        XX, YY = np.meshgrid(X, Y)
        grid = np.vstack((XX.flatten(), YY.flatten())).T
        ZZ = dist.pdf(grid).reshape(XX.shape)

        contour = ax.contour(
            XX, YY, ZZ, colors="magenta", levels=num_levels, linewidths=1
        )

        contour_levels = contour.collections

        # Display only the outermost contour level
        outermost_level = contour_levels[1]
        outermost_level.set_edgecolor("limegreen" if i == 0 else "red")

        for j in range(2, num_levels + 1):
            contour_levels[j].set_alpha(0)

        if i == 0:
            lower, upper = np.min(contour.allsegs[1]), np.max(
                contour.allsegs[1]
            )

    ax.plot([lower, upper], [0, 0], c="blue", label="$do(X_1 = x_1)$", lw=1)
    ax.plot(
        [0, 0], [lower, upper], c="blue", label="$do(X_2 = x_2)$", ls="--", lw=1
    )

    ax.legend(loc="upper left")
    ax.set_xlabel("Feature 1 ($x_1$)")
    ax.set_ylabel("Feature 2 ($x_1$)")
    prettify(ax=ax)
    save_figure(fig, savedir, "manifold")


if __name__ == "__main__":
    logger = create_logger(__name__)
    logger.info("Running manifold plot analysis")

    savedir = Path(__file__).parent / "docs/sim09-manifold-plot"
    os.makedirs(savedir, exist_ok=True)

    np.random.seed(11)
    plot_manifold(savedir)
