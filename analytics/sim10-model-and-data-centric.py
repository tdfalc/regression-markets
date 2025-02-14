import os
from pathlib import Path
from itertools import cycle
from typing import Tuple

import numpy as np
from scipy import stats
from joblib import delayed, Parallel
import matplotlib.pyplot as plt
import pandas as pd
from tfds.plotting import prettify, use_tex


from regression_markets.market.data import BatchData
from regression_markets.market.task import BayesianLinearRegression
from regression_markets.market.mechanism import BatchMarket
from regression_markets.market.policy import NllShapleyPolicy
from regression_markets.common.log import create_logger
from analytics.helpers import save_figure


def build_market_data(
    coefficients: np.ndarray,
    autocorrelations: np.ndarray,
    error_covariance: np.ndarray,
    sample_size: int,
    test_frac: float = 0.5,
) -> BatchData:
    def build_var_process(coefficients: np.ndarray, autocorrelations: np.ndarray):
        var = np.eye(len(coefficients) + 1)
        np.fill_diagonal(var, autocorrelations)
        var[0, 1:] = coefficients
        return var

    var = build_var_process(coefficients, autocorrelations)
    data = np.zeros((sample_size + 1, len(coefficients) + 1))
    errors = np.random.multivariate_normal(
        np.zeros(len(coefficients) + 1), error_covariance, sample_size + 1
    )
    for t in range(1, len(data)):
        data[t] = np.dot(var, data[t - 1]) + errors[t]

    def time_shift(array: np.ndarray, lag: int):
        result = np.empty_like(array)
        if lag > 0:
            result[:lag] = np.nan
            result[lag:] = array[:-lag]
        elif lag < 0:
            result[lag:] = np.nan
            result[:lag] = array[-lag:]
        else:
            result[:] = array
        return result.reshape(-1, 1)

    central_agent_lags = time_shift(data[:, 0], 1)
    support_agent_lags = np.hstack(
        [time_shift(data[:, i], 1) for i in range(1, len(coefficients) + 1)]
    )

    return BatchData(
        dummy_feature=np.ones((len(data) - 1, 1)),
        central_agent_features=central_agent_lags[1:],
        support_agent_features=support_agent_lags[1:],
        target_signal=data[1:, 0:1],
        polynomial_degree=1,
        test_frac=test_frac,
    )


def run_experiment(
    coefficients: np.ndarray,
    autocorrelations: np.ndarray,
    error_covariance: np.ndarray,
    sample_size: int,
    num_samples: int,
    noise_variance: float,
    regularization: float = 1e-32,
    test_frac: float = 0.5,
) -> Tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float]]:
    def _one_sample():
        market_data = build_market_data(
            coefficients,
            autocorrelations,
            error_covariance,
            sample_size,
            test_frac=test_frac,
        )
        num_seller_features = market_data.num_support_agent_features

        # Calculate allocations for observational and interventional designs
        allocations = {}
        for observational in (True, False):
            task = BayesianLinearRegression(
                noise_variance=noise_variance, regularization=regularization
            )
            market = BatchMarket(
                market_data, regression_task=task, observational=observational
            )
            output = market.run(NllShapleyPolicy)
            allocations[observational] = output["train"]["allocations"]

        # Intervene on the features and calculate the difference in the loss
        indices_gc = np.arange(market_data.X_train.shape[1])
        posterior = task.get_posterior(indices_gc)
        loss_gc = task.calculate_loss(
            market_data.X_train[:, indices_gc],
            market_data.y_train,
            posterior,
            noise_variance,
        )
        loss_diff = np.zeros(num_seller_features)
        for i, agent in enumerate(market_data.active_agents):
            indices = np.setdiff1d(indices_gc, agent)
            posterior_mean = posterior.mean[indices].copy()
            posterior_covariance = posterior.cov[np.sort(indices), :][:, indices].copy()
            loss = task.calculate_loss(
                market_data.X_train[:, indices],
                market_data.y_train,
                stats.multivariate_normal(posterior_mean, posterior_covariance),
                noise_variance,
            )
            loss_diff[i] = np.abs(loss - loss_gc)
        loss_diff /= loss_diff.sum()

        return allocations[True], allocations[False], loss_diff

    allocations_obs, allocations_int, loss_diff = list(
        zip(*Parallel(n_jobs=-1)(delayed(_one_sample)() for _ in range(num_samples)))
    )
    return (
        np.vstack(allocations_obs),
        np.vstack(allocations_int),
        np.vstack(loss_diff),
    )


def plot_results(
    allocations_obs: np.ndarray,
    allocations_int: np.ndarray,
    loss_diff: np.ndarray,
) -> None:
    means = pd.DataFrame(
        {
            "Objective Increase": loss_diff.mean(axis=0),
            "$\pi$ (Model-Centric)": allocations_int.mean(axis=0),
            "$\pi$ (Data-Centric)": allocations_obs.mean(axis=0),
        }
    )

    fig, ax = plt.subplots(dpi=600, figsize=(3.6, 3.2))
    for i, col in enumerate(means.columns):
        ax.bar(
            np.arange(2) + i * 0.2,
            means[col],
            label=means.columns[i],
            capsize=5,
            width=0.2,
            edgecolor="k",
        )

    ax.set_xticks([0.25, 1.25])
    ax.set_xticklabels(["$a$", "$b$"])

    ax.minorticks_off()
    ax.legend()
    ax.set_ylabel("Allocation")
    ax.set_xlabel("Support Agent")
    save_figure(fig, savedir, "allocation")


if __name__ == "__main__":
    logger = create_logger(__name__)
    logger.info("Running model and data centric analysis")

    use_tex()

    savedir = Path(__file__).parent / "docs/sim10-model-and-data-centric"
    os.makedirs(savedir, exist_ok=True)

    test_frac = 0.5
    regularization = 1e-5
    num_samples = 1000
    sample_size = 1500

    coefficients = np.array([0.9, 0])
    autocorrelations = np.array([0.12, 0.54, 0.49])
    error_covariance = np.array(
        [
            [1, 0, 0],
            [0, 1, 0.96],
            [0, 0.96, 1],
        ]
    )
    noise_variance = 1

    allocations_obs, allocations_int, loss_diff = run_experiment(
        coefficients=coefficients,
        autocorrelations=autocorrelations,
        error_covariance=error_covariance,
        sample_size=sample_size,
        num_samples=num_samples,
        noise_variance=noise_variance,
        regularization=regularization,
        test_frac=test_frac,
    )

    plot_results(allocations_obs, allocations_int, loss_diff)
