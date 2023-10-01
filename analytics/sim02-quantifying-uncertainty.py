from pathlib import Path
import os

import numpy as np
from matplotlib import cm
import matplotlib as mpl
from matplotlib import pyplot as plt
from scipy.stats._multivariate import multivariate_normal_frozen as mvn_frozen

from market.task import BayesianLinearRegression
from analytics.helpers import save_figure, add_dummy, set_plot_style
from common.log import create_logger


def make_regression(
    coefficients: np.ndarray, noise_variance: float, sample_size: int
):
    size = (sample_size, 1)
    X = add_dummy(np.random.uniform(size=size) * 2 - 1)
    noise = np.random.normal(scale=np.sqrt(noise_variance), size=size)
    y = X @ coefficients + noise
    return X, y


def plot_posterior(
    ax: mpl.pyplot.axis, distribution: mvn_frozen, coefficients: np.ndarray
):
    resolution = 1000
    grid_x = grid_y = np.linspace(-1, 1, resolution)
    grid_flat = np.dstack(np.meshgrid(grid_x, grid_y)).reshape(-1, 2)
    densities = distribution.pdf(grid_flat).reshape(resolution, resolution)
    ax.imshow(
        densities[::-1, :],
        cmap=cm.rainbow,
        aspect="auto",
        extent=(-1, 1, -1, 1),
    )
    ax.axvline(x=coefficients[0], ls="--", c="black", lw=1)
    ax.axhline(y=coefficients[1], ls="--", c="black", lw=1)
    ax.set_xlabel(r"$w_0$")
    ax.set_ylabel(r"$w_1$")
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])


def plot_posterior_samples(
    ax: mpl.pyplot.axis,
    X: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_samples: np.ndarray,
    num_samples: int,
):
    labels = ["Posterior Samples"] + [None] * (num_samples - 1)
    ax.plot(X_test[:, 1], y_samples, c="C1", label=labels)
    ax.plot(X_test[:, 1], y_test, c="k", ls="--", label="Ground Truth")
    ax.scatter(X[:, 1], y, marker="o", c="k", s=20)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    plt.legend()


def plot_predictive_uncertainty(
    ax: mpl.pyplot.axis,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred_mean: np.ndarray,
    y_pred_variance: np.ndarray,
):
    ax.plot(X_test[:, 1].ravel(), y_test.ravel(), c="k", ls="--")
    ax.plot(X_test[:, 1], y_pred_mean.ravel(), c="red", label="Predictive Mean")

    for num_sdevs in [1, 2]:
        ax.fill_between(
            X_test[:, 1].ravel(),
            y_pred_mean.ravel() + num_sdevs * np.sqrt(y_pred_variance.ravel()),
            y_pred_mean.ravel() - num_sdevs * np.sqrt(y_pred_variance.ravel()),
            alpha=0.5,
            color="C0",
            label=f"±{num_sdevs}$\sigma$",
        )
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.legend()


def main():
    logger = create_logger(__name__)
    logger.info("Running quantifying uncertainty analysis")

    set_plot_style()

    savedir = Path(__file__).parent / "docs/sim02-quantifying-uncertainty"
    os.makedirs(savedir, exist_ok=True)

    coefficients = np.array([[-0.3], [0.4]])
    noise_variance = 0.04
    regularization = 1
    num_samples = 5

    experiments = [
        {"sample_size": 1},
        {"sample_size": 3},
        {"sample_size": 40},
    ]

    fig = plt.figure(figsize=(15, 10))
    plt.subplots_adjust(hspace=0.6, wspace=0.3)

    for i, experiment in enumerate(experiments):
        X, y = make_regression(coefficients, noise_variance, **experiment)
        task = BayesianLinearRegression(
            noise_variance=noise_variance, regularization=regularization
        )
        indicies = np.arange(X.shape[1])
        task.update_posterior(X, y, indicies)
        posterior = task.get_posterior(indicies)
        posterior_samples = posterior.rvs(num_samples)

        X_test = add_dummy(np.linspace(-1, 1, 100).reshape(-1, 1))
        y_test = X_test @ coefficients
        y_samples = X_test.dot(posterior_samples.T)
        y_pred_mean, y_pred_variance = task._predict(
            X_test, posterior, noise_variance
        )

        ax1 = plt.subplot(len(experiments), 3, i * 3 + 1)
        plot_posterior(ax1, posterior, coefficients)

        ax2 = plt.subplot(len(experiments), 3, i * 3 + 2)
        plot_posterior_samples(
            ax2, X, y, X_test, y_test, y_samples, num_samples
        )

        if i > 0:
            ax2.get_shared_y_axes().join(ax2, ax3)

        ax3 = plt.subplot(len(experiments), 3, i * 3 + 3)
        plot_posterior_samples(
            ax3, X, y, X_test, y_test, y_samples, num_samples
        )
        plot_predictive_uncertainty(
            ax3, X_test, y_test, y_pred_mean, y_pred_variance
        )

        ax3.get_shared_y_axes().join(ax3, ax2)

    save_figure(fig, savedir, "bayesian_updates")


if __name__ == "__main__":
    np.random.seed(42)
    main()
