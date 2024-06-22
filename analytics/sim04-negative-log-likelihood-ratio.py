from typing import Sequence, Callable, Dict
from pathlib import Path
import os
from collections import defaultdict
from joblib import Parallel, delayed
from itertools import cycle

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from regression_markets.market.task import (
    BayesianLinearRegression,
    MaximumLikelihoodLinearRegression,
    Task,
)
from analytics.helpers import save_figure, add_dummy, set_style
from regression_markets.common.log import create_logger
from regression_markets.common.utils import tqdm_joblib, cache


def build_data(
    test_size: float,
    interpolant_function: Callable,
    additive_noise_function: Callable,
    heteroskedasticity_function: Callable,
) -> Callable:
    def _split_data(X, y):
        train_size = int(len(X) - test_size)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        return X_train, X_test, y_train, y_test

    def _build_data(sample_size: int):
        mean, cov, size = np.zeros(3), np.eye(3), sample_size + test_size
        X = add_dummy(np.random.multivariate_normal(mean, cov, size=size))
        noise = additive_noise_function(sample_size)
        y = interpolant_function(X) + noise * heteroskedasticity_function(X)
        return _split_data(X, y)

    return _build_data


def calculate_loss(
    task: Task,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    **kwargs
) -> float:
    task = task(**kwargs)
    indices = np.arange(X_train.shape[1])
    task.update_posterior(X_train, y_train, indices)
    posterior = task.get_posterior(indices)
    noise_variance = task.get_noise_variance(indices)
    return task.calculate_loss(X_test, y_test, posterior, noise_variance)


def run_experiment(
    sample_sizes: np.ndarray,
    regularization: float,
    noise_variance: float,
    experiments: Sequence[dict],
) -> Dict:
    results = defaultdict(list)
    for sample_size in sample_sizes:
        for title, config in experiments.items():
            sample_size = int(sample_size)
            X_train, X_test, y_train, y_test = config["build_data"](sample_size)

            nll_mle = calculate_loss(
                MaximumLikelihoodLinearRegression,
                X_train,
                X_test,
                y_train,
                y_test,
                noise_variance=noise_variance,
            )

            nll_blr = calculate_loss(
                BayesianLinearRegression,
                X_train,
                X_test,
                y_train,
                y_test,
                regularization=regularization,
                noise_variance=noise_variance,
            )

            results[title].append((nll_blr.mean()) / (nll_mle.mean()))

    return results


def main() -> None:
    logger = create_logger(__name__)
    logger.info("Running negative log likelihood ratio analysis")

    savedir = Path(__file__).parent / "docs/sim04-negative-log-likelihood-ratio"
    os.makedirs(savedir, exist_ok=True)

    set_style()

    coefficients = np.array([[-0.1], [0.3], [0.8], [-0.4]])
    noise_variance = 2
    regularization = 1e-5
    num_samples = 1000

    sample_sizes = np.geomspace(10, 10000, 15)
    test_size = 1000

    # Well-specified model
    interpolant_function = lambda X: X @ coefficients
    additive_noise_function = lambda sample_size: np.random.normal(
        0,
        np.sqrt(noise_variance),
        size=(sample_size + test_size, 1),
    )
    heteroskedasticity_function = lambda X: 1

    # Induced misspecifications
    misspecified_interpolant_function = lambda X: X**2 @ coefficients
    misspecified_additive_noise_function = lambda sample_size: np.random.standard_t(
        df=2, size=(sample_size + test_size, 1)
    )
    misspecified_heteroskedasticity_function = lambda X: X[:, -2:-1] ** 2

    experiments = {
        "Baseline": {
            "build_data": build_data(
                test_size=test_size,
                interpolant_function=interpolant_function,
                additive_noise_function=additive_noise_function,
                heteroskedasticity_function=heteroskedasticity_function,
            ),
        },
        "Interpolant": {
            "build_data": build_data(
                test_size=test_size,
                interpolant_function=misspecified_interpolant_function,
                additive_noise_function=additive_noise_function,
                heteroskedasticity_function=heteroskedasticity_function,
            ),
        },
        "Noise": {
            "build_data": build_data(
                test_size=test_size,
                interpolant_function=misspecified_interpolant_function,
                additive_noise_function=misspecified_additive_noise_function,
                heteroskedasticity_function=heteroskedasticity_function,
            ),
        },
        "Heteroskedasticity": {
            "build_data": build_data(
                test_size=test_size,
                interpolant_function=misspecified_interpolant_function,
                additive_noise_function=misspecified_additive_noise_function,
                heteroskedasticity_function=misspecified_heteroskedasticity_function,
            ),
        },
    }

    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    markers = cycle(["o", "d", ">", "s"])
    cache_location = savedir / "cache"
    os.makedirs(cache_location, exist_ok=True)
    with tqdm_joblib(tqdm(desc="Simulation progress", total=num_samples)) as _:
        results = cache(save_dir=cache_location)(
            lambda: Parallel(n_jobs=-1)(
                delayed(run_experiment)(
                    sample_sizes, regularization, noise_variance, experiments
                )
                for _ in range(num_samples)
            )
        )()

    for name, _ in experiments.items():
        data = 1 - np.vstack([r[name] for r in results]).T.mean(axis=1)
        ax.plot(
            sample_sizes,
            data * 100,
            label=name,
            color="k",
            marker=next(markers),
            markeredgecolor="k",
            markeredgewidth=0.8,
            lw=1,
            markerfacecolor="White",
        )

    ax.legend(framealpha=0)
    ax.set_ylabel(r"Improvement (\%)")
    ax.set_xlabel("Sample Size")
    ax.set_xscale("log")
    ax.set_ylim([-5, 45])

    fig.subplots_adjust(wspace=10)
    save_figure(fig, savedir, "nll_ratio")


if __name__ == "__main__":
    np.random.seed(1)
    main()
