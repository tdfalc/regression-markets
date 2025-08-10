from pathlib import Path
import os

import numpy as np
from matplotlib import cm, pyplot as plt
import matplotlib as mpl
from scipy.stats._multivariate import multivariate_normal_frozen as mvn_frozen
from tfds.plotting import use_tex, prettify
from tqdm import tqdm
from scipy.ndimage import gaussian_filter as smooth

from regression_markets.market.task import (
    BayesianLinearRegression as Blr,
    MaximumLikelihoodLinearRegression as Mle,
)
from analytics.helpers import save_figure, add_dummy, set_style
from regression_markets.common.log import create_logger

use_tex()
set_style()


def plot_posterior(
    ax: mpl.axes.SubplotBase,
    distribution: mvn_frozen,
    coefficients: np.ndarray,
    xlabel: bool = False,
) -> None:
    resolution = 1000
    grid = np.linspace(-1, 1, resolution)
    grid_x, grid_y = np.meshgrid(grid, grid)
    grid_flat = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    densities = distribution.pdf(grid_flat).reshape(resolution, resolution)

    ax.imshow(
        densities[::-1, :],
        cmap=cm.YlGnBu,
        # cmap=cm.viridis_r,
        # alpha=0.5,
        aspect="auto",
        extent=(-1, 1, -1, 1),
    )
    ax.axvline(x=coefficients[0], ls="--", c="black", lw=1)
    ax.axhline(y=coefficients[1], ls="--", c="black", lw=1)
    if xlabel:
        ax.set_xlabel(r"$w_0$")
    else:
        ax.set_xticklabels([])
    ax.set_ylabel(r"$w_1$")
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    prettify(ax=ax, legend=False)


def run_bayesian_updates_experiment(savedir: Path):
    coefficients = np.array([[-0.3], [0.5]])
    noise_variance = 0.07
    regularization = 1
    sample_size = 1000

    X = add_dummy(np.linspace(-1, 1, sample_size).reshape(-1, 1))
    y = X @ coefficients
    y_noisy = y + np.random.normal(
        scale=np.sqrt(noise_variance), size=(sample_size, 1)
    )
    idx = np.random.randint(sample_size, size=sample_size)

    experiments = [{"train_size": 1}, {"train_size": 3}, {"train_size": 20}]

    fig, axs = plt.subplots(3, 2, figsize=(9, 8), width_ratios=[1, 2])

    for i, experiment in enumerate(experiments):
        train_size = experiment["train_size"]
        train_idx = idx[:train_size]

        task = Blr(noise_variance=noise_variance, regularization=regularization)
        X_train, y_train = X[train_idx, :], y_noisy[train_idx]

        indices = np.arange(X_train.shape[1])
        task.update_posterior(X_train, y_train, indices)
        posterior = task.get_posterior(indices)

        ax_post = axs[i, 0]
        plot_posterior(ax_post, posterior, coefficients, xlabel=i == 2)
        ax_post.set_xticks([-1, 0, 1])
        ax_post.set_yticks([-1, 0, 1])
        prettify(ax=ax_post, legend=False)

        ax_pred = axs[i, 1]
        if i < 2:
            ax_pred.set_xticklabels([])
        xs = X[:, 1]
        ax_pred.plot(xs, y, ls="dashed", color="k", lw=1)
        cmap = plt.get_cmap("viridis", 5)
        color = cmap.colors[2]

        ax_pred.set_title(f"$N = {train_size}$")

        y_hats = X @ posterior.rvs(100).T
        mean = y_hats.mean(axis=1)
        sdev = y_hats.std(axis=1)
        ax_pred.plot(xs, mean, c=color, lw=1)
        ax_pred.fill_between(xs, mean + sdev, mean - sdev, fc=color, alpha=0.25)
        ax_pred.plot(xs, mean + sdev, color=color, lw=1)
        ax_pred.plot(xs, mean - sdev, color=color, lw=1)

        ax_pred.scatter(
            X_train[:, 1], y_train, marker="x", s=50, c="r", zorder=3
        )
        ax_pred.set_xlim(-1, 1)
        ax_pred.set_xticks([-1, -0.6, -0.2, 0.2, 0.6, 1])
        if i == 2:
            ax_pred.set_xlabel(r"$x^{(t)}$")
        ax_pred.set_ylabel(r"$y^{(t)}$")
        ax_pred.set_ylim(-1.5, 1.3)
        prettify(ax=ax_pred, legend=False, ticks=False)

    fig.tight_layout()
    save_figure(fig, savedir, "bayesian_updates")


def run_polynomial_fit_experiment(savedir: Path):
    powers = [0, 1, 2, 3]
    fig, axs = plt.subplots(2, 2, figsize=(9, 6), sharey=True, sharex=True)
    regularization = 5e-3
    noise_variance = 0.05
    sample_size = 100

    X = np.linspace(0, 1, sample_size).reshape(-1, 1)
    y = (0.5 + np.sin(2 * np.pi * X[:, 0])).reshape(-1, 1)
    y_noisy = y + np.random.normal(
        0, np.sqrt(noise_variance), size=(sample_size, 1)
    )
    idx_train = np.random.choice(sample_size, size=10, replace=False)

    for i, p in enumerate(powers):
        task = Blr(noise_variance=noise_variance, regularization=regularization)
        cmap = plt.get_cmap("viridis", 5)
        color = cmap.colors[2]

        ax = axs.flatten()[i]
        xs = X[:, 0]
        ax.set_xlim(0, 1)
        ax.plot(xs, y.flatten(), color="k", lw=1, ls="dashed")

        # Create design matrix (Phi) for polynomial features up to degree p
        Phi = add_dummy(np.vstack([X[:, 0] ** l for l in range(p + 1)]).T)
        X_train = X[idx_train, :]
        Phi_train = Phi[idx_train, :]
        y_train = y_noisy[idx_train]

        indices = np.arange(Phi_train.shape[1])
        task.update_posterior(Phi_train, y_train, indices)
        posterior = task.get_posterior(indices)

        y_hats = Phi @ posterior.rvs(100).T
        mean = y_hats.mean(axis=1)
        sdev = y_hats.std(axis=1)

        ax.plot(xs, mean, color=color, lw=1)
        ax.fill_between(xs, mean + sdev, mean - sdev, fc=color, alpha=0.25)
        ax.plot(xs, mean + sdev, color=color, lw=1)
        ax.plot(xs, mean - sdev, color=color, lw=1)
        ax.scatter(
            X_train[:, 0], y_train, marker="x", s=50, color="r", zorder=3
        )
        if i > 1:
            ax.set_xlabel(r"$x^{(t)}$")
        if i % 2 == 0:
            ax.set_ylabel(r"$y^{(t)}$")
        ax.set_title(f"$c={p}$")
        ax.set_ylim(-1, 2)
        # ax.grid(c="#c0c0c0", alpha=0.5, lw=1)
        prettify(ax=ax, legend=False, ticks=False)

    fig.tight_layout()
    save_figure(fig, savedir, "polynomial_fit")


def run_overfitting_experiment(savedir: Path):
    # Experiment parameters
    num_runs = 100
    max_power = 10
    powers = np.arange(0, max_power + 1)
    regularization = 1e-6
    noise_variance = 0.05
    sample_size = 1000

    # Generate clean data
    X = np.linspace(0, 1, sample_size).reshape(-1, 1)
    y = (0.5 + np.sin(2 * np.pi * X[:, 0])).reshape(-1, 1)

    # Initialize results dictionary for losses and predictions
    results = {
        "blr": {"loss": np.zeros((num_runs, len(powers))), "pred": []},
        "nll": {"loss": np.zeros((num_runs, len(powers))), "pred": []},
    }

    fig, (ax_loss, ax_pred) = plt.subplots(
        1, 2, figsize=(9.5, 3.25), width_ratios=[1, 2]
    )

    for run in tqdm(range(num_runs), desc="Runs"):
        # Add noise for current run
        y_noisy = y + np.random.normal(0, np.sqrt(noise_variance), size=y.shape)

        # Select training indices (10 random points) and compute test indices
        idx_train = np.random.randint(len(X), size=15)
        idx_test = np.setdiff1d(np.arange(len(X)), idx_train)

        # Loop over the two model types: 'blr' and 'nll'
        for model in ("blr", "nll"):
            for i, p in enumerate(powers):
                # Select model based on type
                if model == "blr":
                    task = Blr(
                        noise_variance=noise_variance,
                        regularization=regularization,
                    )
                else:
                    task = Mle(noise_variance=noise_variance)

                Phi = np.vstack([X[:, 0] ** l for l in range(p + 1)]).T
                X_train = X[idx_train, :]
                Phi_train = Phi[idx_train, :]
                y_train = y_noisy[idx_train]

                # Update the posterior using training data
                indices = np.arange(Phi_train.shape[1])
                task.update_posterior(Phi_train, y_train, indices)
                posterior = task.get_posterior(indices)

                # Compute loss on the test set and store it
                loss = task.calculate_loss(
                    Phi[idx_test], y_noisy[idx_test], posterior, noise_variance
                )
                results[model]["loss"][run, i] = loss

                # Save predictions only on the first run
                if run == 0:
                    results[model]["pred"].append(
                        task._predict(Phi[idx_test], posterior, noise_variance)
                    )

    cmap = plt.get_cmap("viridis", 5)
    colors = cmap.colors[[2, -1]]

    # Plot loss curves for each model
    for i, model in enumerate(("blr", "nll")):
        mean_loss = smooth(results[model]["loss"].mean(axis=0), sigma=0.5)
        ax_loss.plot(powers, mean_loss, color=colors[i], lw=1, label=model)
        ax_loss.scatter(
            powers,
            mean_loss,
            color=colors[i],
            marker="o",
            facecolor="w",
            zorder=3,
        )
    ax_loss.set_yscale("log")
    ax_loss.set_xlabel("$c$")
    ax_loss.set_ylabel(r"$\mathbb{E}[-\log \hat{y}^{(t)}]$")
    ax_loss.set_xticks(powers[::2])
    prettify(ax=ax_loss, legend=False, legend_loc="upper left")

    # Plot predictions on test data
    X_test = X[idx_test, 0]
    ax_pred.plot(
        X_test,
        y.flatten()[idx_test],
        color="k",
        ls="dashed",
        lw=1,
        label="Truth",
    )
    for i, model in enumerate(("blr", "nll")):
        pred_mean, pred_var = results[model]["pred"][-1]
        pred_mean = pred_mean.flatten()
        pred_sdev = np.sqrt(pred_var.flatten())
        ax_pred.plot(
            X_test,
            pred_mean,
            color=colors[i],
            ls="solid",
            lw=1,
            zorder=2,
        )
        ax_pred.fill_between(
            X_test,
            pred_mean + pred_sdev,
            pred_mean - pred_sdev,
            facecolor=colors[i],
            alpha=0.25,
        )
        ax_pred.plot(X_test, pred_mean + pred_sdev, color=colors[i], lw=1)
        ax_pred.plot(X_test, pred_mean - pred_sdev, color=colors[i], lw=1)
        ax_pred.scatter(
            X_train[:, 0], y_train, marker="x", s=50, color="r", zorder=3
        )
    ax_pred.set_ylim(-1, 2)
    ax_pred.set_xlabel(r"$x^{(t)}$")
    ax_pred.set_ylabel(r"$y^{(t)}$")
    ax_pred.set_xlim(left=0, right=1)
    prettify(ax=ax_pred, legend=False)

    fig.tight_layout()
    save_figure(fig, savedir, "overfitting")


def main() -> None:
    logger = create_logger(__name__)
    logger.info("Running bayesian linear regression analysis")

    savedir = Path(__file__).parent / "docs/sim02-blr"
    os.makedirs(savedir, exist_ok=True)

    set_style()

    run_bayesian_updates_experiment(savedir)
    run_polynomial_fit_experiment(savedir)
    run_overfitting_experiment(savedir)


if __name__ == "__main__":
    np.random.seed(123)
    main()
