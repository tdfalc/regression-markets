import os
from pathlib import Path
from itertools import cycle

import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
import pandas as pd

from common.log import create_logger
from analytics.helpers import save_figure, get_pyplot_colors


def plot_manifold(savedir: Path):
    rhos = [0, 0.9]
    colors = cycle(get_pyplot_colors()[:])

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
        contour = ax.contour(XX, YY, ZZ, colors=color, levels=num_levels, linewidths=2)
        contour_levels = contour.collections

        # Display only the outermost contour level
        outermost_level = contour_levels[1]
        outermost_level.set_edgecolor(color)

        for j in range(2, num_levels + 1):
            contour_levels[j].set_alpha(0)

        if i == 0:
            lower, upper = np.min(contour.allsegs[1]), np.max(contour.allsegs[1])

        # outermost_level.set_offset_position("data")
        # print(outermost_level.get_offsets())

        level_segs = contour.allsegs[1][0]
        x, y = level_segs[:, 0], level_segs[:, 1]
        pd.DataFrame({"x": x, "y": y}).to_csv(
            savedir / f"data{i}.txt", index=False, sep="\t"
        )

    color = next(colors)
    l1 = ax.plot([lower, upper], [0, 0], c=color, label="$do(X_1 = x_1)$", lw=2)
    l2 = ax.plot(
        [0, 0], [lower, upper], c=color, label="$do(X_2 = x_2)$", ls="--", lw=2
    )

    l1_data = np.array(l1[0].get_data()).T
    l2_data = np.array(l2[0].get_data()).T
    pd.DataFrame({"x": l1_data[:, 0], "y": l1_data[:, 1]}).to_csv(
        savedir / f"l1data{i}.txt", index=False, sep="\t"
    )
    pd.DataFrame({"x": l2_data[:, 0], "y": l2_data[:, 1]}).to_csv(
        savedir / f"l2data{i}.txt", index=False, sep="\t"
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
