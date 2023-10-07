import os
from pathlib import Path
from itertools import cycle

import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

from common.log import create_logger
from analytics.helpers import save_figure, julia_colors


def plot_manifold(savedir: Path):
    rhos = [0, 0.9]
    colors = cycle(julia_colors()[:])

    vmin, vmax = -3, 3
    resolution = 100

    fig, ax = plt.subplots(figsize=(3.6, 3.2), dpi=600)

    for i, rho in enumerate(rhos):
        mean = np.zeros(2)
        covariance = [[1, rho], [rho, 1]]
        dist = stats.multivariate_normal(mean=mean, cov=covariance)

        X = np.linspace(vmin, vmax, resolution)
        Y = np.linspace(vmin, vmax, resolution)
        XX, YY = np.meshgrid(X, Y)
        grid = np.vstack((XX.flatten(), YY.flatten())).T
        ZZ = dist.pdf(grid).reshape(XX.shape)

        color = next(colors)

        num_levels = 10
        contour = ax.contour(
            XX, YY, ZZ, colors=color, levels=num_levels, linewidths=2
        )
        contour_levels = contour.collections

        # Display only the outermost contour level
        outermost_level = contour_levels[1]
        outermost_level.set_edgecolor(color)

        for j in range(2, num_levels + 1):
            contour_levels[j].set_alpha(0)

        if i == 0:
            lower, upper = np.min(contour.allsegs[1]), np.max(
                contour.allsegs[1]
            )

    color = next(colors)
    ax.plot([lower, upper], [0, 0], c=color, label="$do(X_1 = x_1)$", lw=2)
    ax.plot(
        [0, 0], [lower, upper], c=color, label="$do(X_2 = x_2)$", ls="--", lw=2
    )
    ax.legend(loc="upper left")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    save_figure(fig, savedir, "manifold")


if __name__ == "__main__":
    logger = create_logger(__name__)
    logger.info("Running manifold plot analysis")

    savedir = Path(__file__).parent / "docs/sim09-manifold-plot"
    os.makedirs(savedir, exist_ok=True)

    np.random.seed(11)
    plot_manifold(savedir)
