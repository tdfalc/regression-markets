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
from tfds.plotting import use_tex, prettify
import matplotlib.colors as mcolors

from regression_markets.market.task import BayesianLinearRegression
from regression_markets.market.data import BatchData
from regression_markets.market.mechanism import BatchMarket
from regression_markets.market.policy import NllShapleyPolicy
from analytics.helpers import save_figure, build_input, set_style
from regression_markets.common.log import create_logger

set_style()
use_tex()


def make_regression(
    coefficients: np.ndarray, noise_variance: float, sample_size: int
) -> Tuple[np.ndarray[float], np.ndarray[float]]:
    num_features = len(coefficients) - 1
    X = build_input(sample_size, num_features)
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
        cmap="YlGnBu",  # cm.jet
        aspect="auto",
        extent=(-1, 1, -1, 1),
    )
    ax.axvline(x=coefficients[2], ls="--", c="black", lw=1)
    ax.axhline(y=coefficients[3], ls="--", c="black", lw=1)
    ax.set_xlabel(r"$w_2$")
    ax.set_ylabel(r"$w_3$")
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])


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
        bootstraps, color=color, alpha=0.25, histtype="stepfilled", label=label
    )
    ax.hist(bootstraps, color=color, histtype="step", label=label, lw=1.5)
    ax.set_xlabel(r"$-\log p(y_t \vert x_t)$")
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
    logger.info("Running quantifying uncertainty analysis")

    savedir = Path(__file__).parent / "docs/sim03-quantifying-uncertainty"
    os.makedirs(savedir, exist_ok=True)

    set_style()

    coefficients = np.array([[-0.11], [0.31], [0.08], [0.65]])
    noise_variance = 1 / 3.31
    regularization = 1e-6
    test_size = 100
    train_payment = 0.005

    X, y = make_regression(coefficients, noise_variance, sample_size=100)
    X_test, y_test = make_regression(coefficients, 0, sample_size=test_size)

    experiments = [
        {"sample_size": 5},
        {"sample_size": 10},
        {"sample_size": 50},
    ]

    fig = plt.figure(figsize=(8, 6.8))
    axs = np.empty((len(experiments), 3), dtype=mpl.axes.SubplotBase)

    for i, experiment in enumerate(experiments):
        sample_size = experiment["sample_size"]

        X_train, y_train = X[:sample_size], y[:sample_size]
        X_test, y_test = X[:sample_size], y[:sample_size]

        task = BayesianLinearRegression(
            noise_variance=noise_variance, regularization=regularization
        )

        indicies = np.arange(X_train.shape[1])
        task.update_posterior(X_train, y_train, indicies)
        posterior_grand_coalition = task.get_posterior(indicies)

        posterior_support_agents = stats.multivariate_normal(
            posterior_grand_coalition.mean[-2:],
            posterior_grand_coalition.cov[:, -2:][-2:, :],
        )
        axs[i, 0] = plt.subplot(len(experiments), 3, i * 3 + 1)
        plot_posterior(axs[i, 0], posterior_support_agents, coefficients)

        task.update_posterior(X_train, y_train, [0, 1])
        posterior_buyer = task.get_posterior([0, 1])

        y_pred_grand_coalition = task._predict(
            X_test, posterior_grand_coalition, noise_variance
        )
        y_pred_buyer = task._predict(
            X_test[:, [0, 1]], posterior_buyer, noise_variance
        )

        axs[i, 1] = plt.subplot(len(experiments), 3, i * 3 + 2)
        if i > 0:
            axs[i - 1, 1].sharex(axs[i, 1])
            axs[i - 1, 1].sharey(axs[i, 1])

        colors = plt.get_cmap("viridis", 5).colors
        element1 = plot_predictive_uncertainty(
            axs[i, 1],
            y_test,
            *y_pred_buyer,
            color=mcolors.to_hex(colors[1]),
            label="Without Market" if i == 0 else None,
        )
        element2 = plot_predictive_uncertainty(
            axs[i, 1],
            y_test,
            *y_pred_grand_coalition,
            color=mcolors.to_hex(colors[-2]),
            label="With Market" if i == 0 else None,
        )
        if i == 0:
            axs[i, 1].legend(
                handles=[element1, element2],
                frameon=False,
                fontsize=12,
                loc="upper right",
                bbox_to_anchor=[1.05, 1],
            )

        axs[i, 2] = plt.subplot(len(experiments), 3, i * 3 + 3)
        if i > 0:
            axs[i - 1, 2].sharey(axs[i, 2])

        market_data = BatchData(
            X_train[:, [0]], X_train[:, [1]], X_train[:, 2:], y_train
        )
        market_output = BatchMarket(
            market_data, task, train_payment=train_payment
        ).run(NllShapleyPolicy)
        plot_payments(axs[i, 2], market_output, color="k")

    for ax in axs.flatten():
        prettify(ax=ax, legend=False)

    fig.tight_layout()
    save_figure(fig, savedir, "market_outcomes")


if __name__ == "__main__":
    np.random.seed(123)
    main()
