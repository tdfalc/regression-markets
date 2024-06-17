from typing import Callable, Tuple
from pathlib import Path
import os

import numpy as np
from scipy import stats
from matplotlib import cm
import matplotlib as mpl
from matplotlib import pyplot as plt
from scipy.stats._multivariate import multivariate_normal_frozen as mvn_frozen

from common.log import create_logger
from analytics.helpers import save_figure


def make_regression(
    coefficients: np.ndarray,
    noise_variance: float,
    num_samples: int,
    sample_size: int,
    correlation: float,
) -> Tuple[np.ndarray[float], np.ndarray[float]]:
    mean, covaraince = [0, 0], [[1, correlation], [correlation, 1]]
    size = (num_samples, sample_size)
    X = np.random.multivariate_normal(mean, covaraince, size=size)
    noise = np.random.normal(scale=np.sqrt(noise_variance), size=(*size, 1))
    y = X @ coefficients + noise
    return X, y


def plot_density(
    ax: mpl.pyplot.axis,
    distribution: mvn_frozen,
    vmin: float,
    vmax: float,
    xopt: float,
    yopt: float,
    resolution: int,
    **kwargs
) -> None:
    grid_x = grid_y = np.linspace(vmin, vmax, resolution)
    grid_flat = np.dstack(np.meshgrid(grid_x, grid_y)).reshape(-1, 2)
    densities = distribution.pdf(grid_flat).reshape(resolution, resolution)
    ax.imshow(
        densities[:, ::-1],
        extent=(vmin, vmax, vmin, vmax),
        aspect="auto",
        cmap=cm.rainbow,
        **kwargs,
    )
    ax.axvline(x=xopt, ls="--", c="k", lw=1)
    ax.axhline(y=yopt, ls="--", c="k", lw=1)
    ax.set_xlabel(r"$w_0$")
    ax.set_ylabel(r"$w_1$")


def get_surface(
    z_function: Callable, vmin: float, vmax: float, resolution: int
) -> Tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float]]:
    X = np.linspace(vmin, vmax, resolution)
    Y = np.linspace(vmin, vmax, resolution)
    XX, YY = np.meshgrid(X, Y)
    grid = np.vstack((YY.flatten(), XX.flatten())).T
    ZZ = z_function(grid.T).reshape(XX.shape)
    return XX, YY, ZZ


def plot_image(
    ax: mpl.pyplot.axis,
    z_function: Callable,
    vmin: float,
    vmax: float,
    resolution: int = 100,
) -> None:
    _, _, ZZ = get_surface(z_function, vmin, vmax, resolution)
    _ = ax.imshow(ZZ[:, ::-1], cmap=cm.rainbow, aspect="auto")
    ax.set_xticks([0, len(ZZ) / 2, len(ZZ) - 1])
    ax.set_xticklabels([vmin, (vmin + vmax) / 2, vmax])
    ax.set_yticks([0, len(ZZ) / 2, len(ZZ) - 1])
    ax.set_yticklabels([vmin, (vmin + vmax) / 2, vmax])
    ax.set_zorder(-1)
    ax.axvline(x=len(ZZ) / 2, ls="--", c="k", lw=1.2)
    ax.axhline(y=len(ZZ) / 2, ls="--", c="k", lw=1.2)


def plot_contours(
    ax: mpl.pyplot.axis,
    z_function: Callable,
    vmin: float,
    vmax: float,
    filled: bool = False,
) -> None:
    XX, YY, ZZ = get_surface(z_function)
    if filled:
        _ = ax.contourf(XX, YY, ZZ, cmap=cm.rainbow, levels=5)
    else:
        ax.contour(XX, YY, ZZ, cmap=cm.rainbow, levels=5)
    ax.set_xticks([vmin, (vmin + vmax) / 2, vmax])
    ax.set_yticks([vmin, (vmin + vmax) / 2, vmax])
    ax.set_zorder(-1)
    ax.axvline(x=(vmin + vmax) / 2, ls="--", c="k", lw=1)
    ax.axhline(y=(vmin + vmax) / 2, ls="--", c="k", lw=1)


def main() -> None:
    logger = create_logger(__name__)
    logger.info("Running visualizing uncertainty analysis")

    savedir = Path(__file__).parent / "docs/sim01-visualizing-uncertainty"
    os.makedirs(savedir, exist_ok=True)

    coefficients = np.array([[0.1], [0.1]])
    num_samples = 100
    noise_variance = 0.5

    experiments = {
        "baseline": {"sample_size": 500, "correlation": 0},
        "low_sample_size": {"sample_size": 50, "correlation": 0},
        "high_correlation": {"sample_size": 500, "correlation": 0.995},
    }

    fig, axs = plt.subplots(1, 3, figsize=(8, 2.5), sharey=True, sharex=True)

    for ax, (_, experiment_config) in zip(axs.flatten(), experiments.items()):
        X, y = make_regression(
            coefficients, noise_variance, num_samples, **experiment_config
        )

        psuedo_inverse = np.linalg.inv(X.transpose(0, 2, 1) @ X) @ X.transpose(0, 2, 1)
        mle_estimates = (psuedo_inverse @ y).reshape(-1, 2)
        mean = np.mean(mle_estimates, axis=0)
        covariance = np.cov(mle_estimates.T)
        distribution = stats.multivariate_normal(mean.ravel(), covariance)

        plot_density(
            ax,
            distribution,
            vmin=coefficients[0][0] - 0.2,
            vmax=coefficients[1][0] + 0.2,
            xopt=coefficients[0][0],
            yopt=coefficients[1][0],
            resolution=1000,
        )

    save_figure(fig, savedir, "coefficient_uncertainty")


if __name__ == "__main__":
    np.random.seed(42)
    main()
