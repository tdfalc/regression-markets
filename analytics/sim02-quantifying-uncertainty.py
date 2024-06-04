from pathlib import Path
import os
from typing import Dict

import numpy as np
from matplotlib import cm
import matplotlib as mpl
from matplotlib import pyplot as plt
from scipy.stats._multivariate import multivariate_normal_frozen as mvn_frozen
from scipy import stats
from matplotlib.patches import Patch

from market.task import BayesianLinearRegression
from market.data import BatchData
from market.mechanism import BatchMarket
from market.policy import NllShapleyPolicy
from analytics.helpers import save_figure, add_dummy, get_julia_colors
from common.log import create_logger


def make_regression(coefficients: np.ndarray, noise_variance: float, sample_size: int):
    X = add_dummy(
        np.random.multivariate_normal(np.zeros(3), np.eye(3), size=sample_size)
    )

    # X = add_dummy(np.random.uniform(size=(sample_size, 3)) * 2 - 1)

    y = X @ coefficients + np.random.normal(
        scale=np.sqrt(noise_variance), size=(sample_size, 1)
    )

    return X, y


def plot_posterior(
    ax: mpl.axes.SubplotBase,
    distribution: mvn_frozen,
    coefficients: np.ndarray,
):
    resolution = 1000
    grid_x = grid_y = np.linspace(-1, 1, resolution)
    grid_flat = np.dstack(np.meshgrid(grid_x, grid_y)).reshape(-1, 2)
    densities = distribution.pdf(grid_flat).reshape(resolution, resolution)

    ax.imshow(
        densities[::-1, :],
        cmap=cm.rainbow,  # cm.jet
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
):
    nll = -stats.norm.logpdf(
        y_test,
        loc=y_pred_mean,
        scale=y_pred_variance**0.5,
    )

    import seaborn as sns

    # sns.kdeplot(
    #     nll.flatten(),
    #     ax=ax,
    #     color=color,
    #     # bins=30,
    # ).set(xlim=(0))
    # ax.set_xlim([0, 5])

    def bs(x, num_samples=1000):
        n = len(x)

        # Generate an array of shape (num_samples, n) with bootstrap samples
        bootstrap_samples = np.random.choice(x, (num_samples, n), replace=True)

        # Calculate the mean of each bootstrap sample along the axis 1
        bootstrap_means = np.mean(bootstrap_samples, axis=1)

        # Calculate the Monte Carlo mean (mean of the bootstrap means)

        return bootstrap_means

    bootstraps = bs(nll.flatten())

    # sns.kdeplot(
    #     nll.flatten(),
    #     ax=ax,
    #     color=color,
    #     # bins=30,
    # ).set(xlim=(0))
    # ax.set_xlim([0, 5])

    ax.hist(
        bootstraps,
        color=color,
        alpha=0.4,
        histtype="stepfilled",
        label=label,
        # bins=10,
    )
    ax.hist(
        bootstraps,
        color=color,
        histtype="step",
        label=label,
        lw=1.5,
        # bins=10,
    )

    ax.set_xlabel("Negative Log Likelihood")
    ax.set_ylabel("Count")

    legend_element = Patch(
        facecolor=color + "44", edgecolor=color, label=label, linewidth=1.5
    )
    return legend_element


def plot_payments(
    ax: mpl.axes.SubplotBase,
    market_output: Dict,
    sample_size: int,
    color: str,
):
    payments = market_output["train"]["payments"] / 1
    # for i in range(1, sample_size + 1):
    #     ax.scatter(
    #         [1, 2],
    #         i * payments,
    #         color=color,
    #         marker="_",
    #         lw=0.6,
    #         s=500,
    #     )
    #     ax.scatter(
    #         [0],
    #         i * -payments.sum(),
    #         color=color,
    #         marker="_",
    #         lw=0.6,
    #         s=500,
    #     )
    print(payments.shape)
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


def main():
    logger = create_logger(__name__)
    logger.info("Running quantifying uncertainty analysis")

    savedir = Path(__file__).parent / "docs/sim02-quantifying-uncertainty"
    os.makedirs(savedir, exist_ok=True)

    coefficients = np.array([[-0.11], [0.31], [0.08], [0.65]])
    noise_variance = 1 / 1.24
    noise_variance = 0.3
    regularization = 1e-6
    test_size = 100

    X, y = make_regression(coefficients, noise_variance, sample_size=100)
    X_test, y_test = make_regression(coefficients, 0, sample_size=test_size)

    experiments = [
        {"sample_size": 5},
        {"sample_size": 10},
        {"sample_size": 50},
    ]

    fig = plt.figure(figsize=(8, 6.8))
    julia_colors = get_julia_colors()

    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    plt.rc("font", size=12)  # controls default text sizes
    plt.rc("axes", labelsize=12)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=12)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=12)  # fontsize of the tick labels
    plt.rc("legend", fontsize=12)  # legend fontsize

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
        y_pred_buyer = task._predict(X_test[:, [0, 1]], posterior_buyer, noise_variance)

        print("Buyer", y_pred_buyer[0][0], y_pred_buyer[1][0])
        print("GC", y_pred_grand_coalition[0][0], y_pred_grand_coalition[1][0])

        axs[i, 1] = plt.subplot(len(experiments), 3, i * 3 + 2)
        if i > 0:
            # axs[i - 1, 1].get_shared_x_axes().join(axs[i, 1], axs[i - 1, 1])
            # axs[i - 1, 1].get_shared_y_axes().join(axs[i, 1], axs[i - 1, 1])

            axs[i - 1, 1].sharex(axs[i, 1])
            axs[i - 1, 1].sharey(axs[i, 1])

        element1 = plot_predictive_uncertainty(
            axs[i, 1],
            y_test,
            *y_pred_buyer,
            color="#673AB7",
            label="Without Market" if i == 0 else None,
        )
        element2 = plot_predictive_uncertainty(
            axs[i, 1],
            y_test,
            *y_pred_grand_coalition,
            color="#FFB300",
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
        market_output = BatchMarket(market_data, task, train_payment=0.005).run(
            NllShapleyPolicy
        )
        plot_payments(axs[i, 2], market_output, len(X_train), color="k")

    fig.tight_layout()
    save_figure(fig, savedir, "bayesian_updates")


if __name__ == "__main__":
    np.random.seed(123)
    main()
