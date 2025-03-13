from pathlib import Path
import os
from typing import Dict, Tuple

import numpy as np
from matplotlib import cm
import matplotlib as mpl
from matplotlib import pyplot as plt
from scipy.stats._multivariate import multivariate_normal_frozen as mvn_frozen
from scipy import stats
from matplotlib.patches import Patch

from regression_markets.market.task import (
    BayesianLinearRegression,
    MaximumLikelihoodLinearRegression,
)
from regression_markets.market.data import BatchData
from regression_markets.market.mechanism import BatchMarket
from regression_markets.market.policy import NllShapleyPolicy
from analytics.helpers import save_figure, add_dummy, set_style
from regression_markets.common.log import create_logger
from tfds.plotting import use_tex, prettify

use_tex()


def make_regression(
    coefficients: np.ndarray, noise_variance: float, sample_size: int
) -> Tuple[np.ndarray[float], np.ndarray[float]]:
    X = add_dummy(np.linspace(-1, 1, sample_size).reshape(-1, 1))
    y = X @ coefficients + np.random.normal(
        scale=np.sqrt(noise_variance), size=(sample_size, 1)
    )

    return X, y


def plot_posterior(
    ax: mpl.axes.SubplotBase,
    distribution: mvn_frozen,
    coefficients: np.ndarray,
) -> None:
    resolution = 1000
    grid_x = grid_y = np.linspace(-1, 1, resolution)
    grid_flat = np.dstack(np.meshgrid(grid_x, grid_y)).reshape(-1, 2)
    densities = distribution.pdf(grid_flat).reshape(resolution, resolution)

    ax.imshow(
        densities[::-1, :],
        cmap=cm.YlGnBu,  # cm.jet
        aspect="auto",
        extent=(-1, 1, -1, 1),
    )
    ax.axvline(x=coefficients[0], ls="--", c="black", lw=1)
    ax.axhline(y=coefficients[1], ls="--", c="black", lw=1)
    ax.set_xlabel(r"$w_2$")
    ax.set_ylabel(r"$w_3$")
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    prettify(ax=ax, legend=False)


def plot_predictive_uncertainty(
    ax: mpl.axes.SubplotBase,
    y_test: np.ndarray,
    y_pred_mean: np.ndarray,
    y_pred_variance: np.ndarray,
    color: str = "C0",
    label: str = None,
) -> None:
    nll = -stats.norm.logpdf(
        y_test,
        loc=y_pred_mean,
        scale=y_pred_variance**0.5,
    )

    def bootstrap(x, num_samples=1000):
        # Generate an array of shape (num_samples, len(x)) with bootstrap samples
        bootstrap_samples = np.random.choice(
            x, (num_samples, len(x)), replace=True
        )
        # Calculate the mean of each bootstrap sample along the axis 1
        return np.mean(bootstrap_samples, axis=1)

    bootstraps = bootstrap(nll.flatten())

    ax.hist(
        bootstraps, color=color, alpha=0.4, histtype="stepfilled", label=label
    )
    ax.hist(bootstraps, color=color, histtype="step", label=label, lw=1.5)
    ax.set_xlabel("Negative Log Likelihood")
    ax.set_ylabel("Count")
    legend_element = Patch(
        facecolor=color + "44", edgecolor=color, label=label, linewidth=1.5
    )
    return legend_element


def plot_payments(
    ax: mpl.axes.SubplotBase, market_output: Dict, color: str
) -> None:
    payments = market_output["train"]["payments"] / 1

    ax.bar(
        np.arange(3),
        np.append(-payments.sum(), payments),
        color=color,
        alpha=0.5,
        edgecolor="k",
        width=0.4,
    )

    ax.set_xticks([0, 1, 2])
    ax.set_xlim([-0.5, 2.5])
    ax.set_xticklabels(["Central\nAgent", "Owner\n$x_2$", "Owner\n$x_3$"])
    ax.axhline(y=0, color="gray", zorder=0, alpha=0.2)
    ax.set_ylabel("Revenue (EUR)")


def main() -> None:
    logger = create_logger(__name__)
    logger.info("Running bayesian linear regression analysis")

    savedir = Path(__file__).parent / "docs/sim022-bayesian-linear-regression"
    os.makedirs(savedir, exist_ok=True)

    set_style()

    coefficients = np.array([[-0.3], [0.5]])
    noise_variance = 1 / 25
    regularization = 1
    test_size = 100

    X, y = make_regression(coefficients, noise_variance, sample_size=1000)
    X_test, y_test = make_regression(coefficients, 0, sample_size=test_size)

    experiments = [
        {"sample_size": 1},
        {"sample_size": 3},
        {"sample_size": 20},
    ]

    fig, axs = plt.subplots(3, 2, figsize=(7, 5.5), width_ratios=[1, 2])

    c = [
        # "#000000",  # Black
        "#F5793A",  # Orange
        "#4BA6FB",  # Modified blue (original was #85C0F9)
        "#A95AA1",  # Pink
        "#689948",  # Green: not optimal for colour-blind people
    ]

    for i, experiment in enumerate(experiments):
        sample_size = experiment["sample_size"]

        task = BayesianLinearRegression(
            noise_variance=noise_variance, regularization=regularization
        )

        idx = np.random.randint(len(X), size=sample_size)

        X_train, y_train = X[idx, :], y[idx]

        indicies = np.arange(X_train.shape[1])
        task.update_posterior(X_train, y_train, indicies)
        posterior = task.get_posterior(indicies)

        ax = axs[i, 0]
        plot_posterior(ax, posterior, coefficients)
        ax.set_xticks([-1, 0, 1])
        prettify(ax=ax, legend=False)

        ax = axs[i, 1]
        ax.set_ylim(bottom=-1.5, top=1.5)

        samples = posterior.rvs(100)

        y_hats = X @ samples.T

        k = 2
        xs = X[:, 1]
        ax.plot(xs, y_hats.mean(axis=1), ls="--", c=c[k])
        mean, sdev = y_hats.mean(axis=1), y_hats.std(axis=1)
        ax.fill_between(
            xs, mean + sdev, mean - sdev, facecolor=c[k], alpha=0.25
        )
        ax.plot(xs, mean + sdev, color=c[k], lw=1)
        ax.plot(xs, mean - sdev, color=c[k], lw=1)

        ax.scatter(
            X_train[:, 1],
            y_train,
            marker="x",
            s=50,
            c=c[k],
            edgecolor="k",
            zorder=3,
        )
        ax.set_xlim(left=-1, right=1)
        ax.set_xlabel(r"$x_{t, 1}$")
        ax.set_ylabel(r"$y$")
        prettify(ax=ax, legend=False)

        # for _ in range(100):
        #     sample = posterior.rvs(1)
        #     y_hat = X @ sample
        #     ax.plot(X[:, 1], y_hat, c=c[k], alpha=0.05)

    fig.tight_layout()
    save_figure(fig, savedir, "bayesian_updates")

    experiments = [
        {"p": 0},
        {"p": 1},
        {"p": 2},
        {"p": 3},
        # {"p": 4},
        # {"p": 5},
        # {"p": 6},
        # {"p": 7},
        # {"p": 8},
        # {"p": 9},
    ]

    fig, axs = plt.subplots(
        2,
        2,
        # height_ratios=[1] ,
    )
    regularization = 5e-3
    # regularization = 1e-5
    noise_variance = 0.3**2
    mlls = []
    X = add_dummy(np.linspace(0, 1, sample_size).reshape(-1, 1))
    y = (0.5 + np.sin(2 * np.pi * X[:, 1])).reshape(-1, 1)
    idx = np.random.randint(len(X), size=10)
    for i, experiment in enumerate(experiments):

        p = experiment["p"]
        task = BayesianLinearRegression(
            noise_variance=noise_variance, regularization=regularization
        )

        ax = axs.flatten()[i]
        xs = X[:, 1]
        ax.set_xlim(left=0, right=1)
        k = 1
        ax.plot(xs, y.flatten(), color="k", lw=1, ls="dashed")

        X_train, y_train = X[idx, :], y[idx]

        def poly(x, p):
            return x**p

        Phi_train = add_dummy(
            np.vstack([poly(X_train[:, 1], l) for l in np.arange(p + 1)]).T
        )

        Phi = add_dummy(
            np.vstack([poly(X[:, 1], l) for l in np.arange(p + 1)]).T
        )

        indicies = np.arange(Phi_train.shape[1])
        task.update_posterior(Phi_train, y_train, indicies)
        posterior = task.get_posterior(indicies)

        samples = posterior.rvs(100)
        y_hats = Phi @ samples.T
        xs = X[:, 1]
        ax.plot(xs, y_hats.mean(axis=1), ls="--", c=c[k])
        mean, sdev = y_hats.mean(axis=1), y_hats.std(axis=1)
        ax.fill_between(
            xs, mean + sdev, mean - sdev, facecolor=c[k], alpha=0.25
        )
        ax.plot(xs, mean + sdev, color=c[k], lw=1)
        ax.plot(xs, mean - sdev, color=c[k], lw=1)

        # for _ in range(100):
        #     sample = posterior.rvs(1)
        #     y_hat = Phi @ sample
        #     ax.plot(xs, y_hat, c=c[k], alpha=0.05)

        ax.set_title(f"$p={p}$")

        ax.set_ylim(top=2, bottom=-1)
        ax.scatter(
            X_train[:, 1],
            y_train,
            marker="o",
            edgecolor="k",
            s=50,
            c=c[k],
            zorder=3,
        )
        ax.set_xlabel(r"$x_t$")
        ax.set_ylabel(r"$y_t$")
        prettify(ax=ax, legend=False)

    fig.tight_layout()
    save_figure(fig, savedir, "bayesian_updates3")


if __name__ == "__main__":
    # np.random.seed(123)# for exp 1
    np.random.seed(42)
    main()
