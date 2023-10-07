import os
from pathlib import Path
from itertools import cycle

import numpy as np
from scipy import stats
from joblib import delayed, Parallel
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import gaussian_filter

from market.data import BatchData
from market.task import BayesianLinearRegression
from market.mechanism import BatchMarket
from market.policy import NllShapleyPolicy
from common.log import create_logger
from analytics.helpers import (
    save_figure,
    julia_colors,
    conditional_value_at_risk,
)
from common.utils import cache


def build_market_data(
    coefficients: np.ndarray,
    autocorrelations: np.ndarray,
    error_covariance: np.ndarray,
    sample_size: int,
    test_frac: float = 0.5,
):
    def build_var_process(
        coefficients: np.ndarray, autocorrelations: np.ndarray
    ):
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


def _run_experiment(
    coefficients: np.ndarray,
    autocorrelations: np.ndarray,
    error_covariance: np.ndarray,
    sample_size: int,
    num_samples: int,
    noise_variance: float,
    regularization: float = 1e-32,
    test_frac: float = 0.5,
):
    def _one_sample():
        market_data = build_market_data(
            coefficients,
            autocorrelations,
            error_covariance,
            sample_size,
            test_frac=test_frac,
        )
        allocations = {}
        for observational in (True, False):
            task = BayesianLinearRegression(
                noise_variance=noise_variance, regularization=regularization
            )
            market = BatchMarket(
                market_data,
                regression_task=task,
                observational=observational,
                train_payment=0.1,
                test_payment=0.1,
            )
            output = market.run(NllShapleyPolicy)
            allocations[observational] = output["train"]["payments"]

        return allocations[True], allocations[False]

    allocations_obs, allocations_int = list(
        zip(
            *Parallel(n_jobs=-1)(
                delayed(_one_sample)() for _ in range(num_samples)
            )
        )
    )
    return np.vstack(allocations_obs), np.vstack(allocations_int)


def run_experiment(
    correlations: np.ndarray,
    coefficients: np.ndarray,
    autocorrelations: np.ndarray,
    error_covariance: np.ndarray,
    sample_size: int,
    num_samples: int,
    noise_variance: float,
    regularization: float = 1e-32,
    test_frac: float = 0.5,
):
    risk_obs, risk_int = [], []
    for correlation in correlations:
        error_covariance[2, 1] = correlation
        error_covariance[1, 2] = correlation
        allocations_obs, allocations_int = _run_experiment(
            coefficients=coefficients,
            autocorrelations=autocorrelations,
            error_covariance=error_covariance,
            sample_size=sample_size,
            num_samples=num_samples,
            noise_variance=noise_variance,
            regularization=regularization,
            test_frac=test_frac,
        )

        risk_obs.append(
            conditional_value_at_risk(allocations_obs, alpha=0.05, axis=0)
        )
        risk_int.append(
            conditional_value_at_risk(allocations_int, alpha=0.05, axis=0)
        )

    return np.vstack(risk_obs), np.vstack(risk_int)


def plot_results(
    correlations: np.ndarray,
    risk_obs: np.ndarray,
    risk_int: np.ndarray,
    savedir: Path,
):
    fig, ax = plt.subplots(dpi=600, figsize=(3.6, 3.2))
    colors = cycle(julia_colors()[5:])

    offset = 1
    ax.plot(
        correlations[:-offset],
        risk_int[:-offset, 1],
        label="Model-Centric",
        c=next(colors),
        lw=2,
    )
    ax.plot(
        correlations[:-offset],
        risk_obs[:-offset, 1],
        label="Data-Centric",
        c=next(colors),
        lw=2,
    )

    ax.legend()
    ax.set_ylabel("Expected Shortfall (USD)")
    ax.set_xlabel("$\\rho$")
    ax.legend()
    ax.minorticks_off()
    ax.axhline(y=0, c="k", ls="dashed", lw=1.6)
    save_figure(fig, savedir, "risk")


if __name__ == "__main__":
    logger = create_logger(__name__)
    logger.info("Running model and data centric analysis")

    savedir = Path(__file__).parent / "docs/sim11-interventional-risk"
    os.makedirs(savedir, exist_ok=True)

    test_frac = 0.5
    regularization = 1e-5
    num_samples = 5000
    sample_size = 1500

    coefficients = np.array([0.9, 0])
    autocorrelations = np.array([0.12, 0.54, 0.49])
    error_covariance = np.array([[1, 0, 0], [0, 1, 0.96], [0, 0.96, 1]])
    noise_variance = 1
    correlations = np.linspace(0, 0.99999, 50)

    cache_location = savedir / "cache"
    os.makedirs(cache_location, exist_ok=True)
    risk_obs, risk_int = cache(save_dir=cache_location)(
        lambda: run_experiment(
            correlations=correlations,
            coefficients=coefficients,
            autocorrelations=autocorrelations,
            error_covariance=error_covariance,
            sample_size=sample_size,
            num_samples=num_samples,
            noise_variance=noise_variance,
            regularization=regularization,
            test_frac=test_frac,
        )
    )()

    plot_results(correlations, risk_obs, risk_int, savedir)
