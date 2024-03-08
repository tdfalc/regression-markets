from pathlib import Path
import os
from typing import Callable, Sequence
from collections import defaultdict
from itertools import cycle

from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from functools import partial

from common.log import create_logger
from common.utils import cache, tqdm_joblib
from market.task import OnlineBayesianLinearRegression
from market.data import MarketData
from market.mechanism import OnlineMarket
from analytics.helpers import (
    save_figure,
    build_input,
    nested_defaultdict,
    bootstrap_resample,
    conditional_value_at_risk,
    MarketDesigns,
    get_pyplot_colors,
)
from market.policy import (
    NllShapleyPolicy,
    KldCfModShapleyPolicy,
    KldContributionModShapleyPolicy,
)


def build_data(
    sample_size: int,
    num_features: int,
    interpolant_function: Callable,
    additive_noise_function: Callable,
    heteroskedasticity_function: Callable,
):
    def _build_data():
        X = build_input(sample_size, num_features)
        noise = additive_noise_function(sample_size)
        y = interpolant_function(X)
        y += noise * heteroskedasticity_function(X)
        return MarketData(X[:, [0]], X[:, [1]], X[:, 2:], y)

    return _build_data


def simulate_online_market(
    market_data: MarketData,
    market_designs: dict,
    likelihood_flattening: float,
    burn_in: int,
    train_payment: float,
    test_payment: float,
):
    results = {}
    for market_design, config in market_designs.items():
        task = config["task"](**config["kwargs"])
        market = config["market"](
            market_data,
            task,
            likelihood_flattening=likelihood_flattening,
            burn_in=burn_in,
            train_payment=train_payment,
            test_payment=test_payment,
        )
        output = market.run(config["policy"])
        results[market_design] = output

    return results


def parse_results(
    results: Sequence[dict],
    num_simulations: int,
    num_forgetting_factors: int,
    num_sellers: int,
    sample_size: int,
    stages: Sequence[str],
    metrics: Sequence[str],
    market_designs: Sequence[str],
):
    empty_array = lambda: np.zeros(
        (num_forgetting_factors, num_sellers, sample_size - 1)
    )
    parsed_results = nested_defaultdict(4, empty_array)

    for i in range(num_simulations):
        for j in range(num_forgetting_factors):
            for market_design in market_designs:
                for stage in stages:
                    for metric in metrics:
                        parsed_results[i][market_design][stage][metric][j] = results[j][
                            i
                        ][market_design][stage][metric].T

    stacked_results = partial(defaultdict, partial(defaultdict, dict))()
    for market_design in market_designs:
        for stage in ("train", "test"):
            for metric in metrics:
                stacked_results[market_design][stage][metric] = np.stack(
                    [
                        result[market_design][stage][metric]
                        for result in parsed_results.values()
                    ],
                    axis=3,
                )

    return stacked_results


def plot_coefficients(coefficients: np.ndarray, savedir):
    fig, ax = plt.subplots(figsize=(4, 2.5))
    for i, agent in enumerate(range(coefficients.shape[1])):
        ax.plot(coefficients[:, agent], label=f"w{i}")
    ax.legend()
    ax.set_ylabel("Coefficient Value")
    ax.set_xlabel("Time Step")
    save_figure(fig, savedir, "coefficients")


def plot_metric_boostrap(results, savedir, idx):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 4.2), sharey=True, sharex=True)

    pyplot_colors = get_pyplot_colors()
    colors = cycle([pyplot_colors[1], pyplot_colors[2], pyplot_colors[0]])

    metric = "payments"
    for i, (ax, stage) in enumerate(zip((ax1, ax2), ("train", "test"))):
        # ax.ticklabel_format(
        #     axis="both", style="scientific", scilimits=(1, 0), useMathText=True
        # )
        for market_design, market_results in results.items():
            color = next(colors)
            metric_results = np.swapaxes(market_results[stage][metric], 0, -1)
            bootstraps = bootstrap_resample(metric_results, 500, sample_size=50)
            expected_value = bootstraps.mean(axis=0)
            expected_shortfall = conditional_value_at_risk(
                bootstraps, axis=0, alpha=0.05
            )

            for seller in (0,):

                def calculate_mean(metric):
                    return metric.mean(axis=0)[seller, :, idx]

                ev_mean = calculate_mean(expected_value)
                es_mean = calculate_mean(expected_shortfall)

                num_runs = np.arange(len(es_mean)) + 1
                ax.plot(
                    num_runs,
                    ev_mean.cumsum(),
                    color=color,
                    ls="-",
                    lw=1,
                    label=market_design.value if seller == 0 else None,
                    zorder=1,
                )
                ax.axhline(y=0, c="lightgray", zorder=0, lw=1)

                ax.yaxis.set_tick_params(labelbottom=True)
                ax.set_ylabel("Revenue (EUR)")
                if i == 1:
                    ax.set_xlabel("Time Step")

    ax1.legend(framealpha=0, ncol=1)

    save_figure(fig, savedir, "contributions_risk")


def main():
    logger = create_logger(__name__)
    logger.info("Running online market analysis")

    savedir = Path(__file__).parent / "docs/sim07-online-market"
    os.makedirs(savedir, exist_ok=True)

    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    plt.rc("font", size=12)  # controls default text sizes
    plt.rc("axes", labelsize=12)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=12)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=12)  # fontsize of the tick labels
    plt.rc("legend", fontsize=12)  # legend fontsize

    config = {
        "num_simulations": 50,  # 1000,
        "noise_variance": 1,
        "train_payment": 0.95,
        "test_payment": 0.95,
        "regularization": 1e-5,
        "sample_size": 1000,
        "forgetting_factors": [0.94],
        "burn_in": 10,
    }

    sample_size = config["sample_size"]

    experiments = {
        "step_change": np.concatenate(
            (
                np.tile(
                    np.array([0, -0.2, 0.1, 0.3]),
                    (int(sample_size / 2), 1),
                ),
                np.tile(
                    np.array([0, -0.2, 0.6, 0.3]),
                    (
                        sample_size - int(sample_size / 2),
                        1,
                    ),
                ),
            ),
            axis=0,
        ),
        # "smooth": np.hstack(
        #     [
        #         np.linspace(0.3, 0.3, sample_size).reshape(-1, 1),
        #         np.linspace(-0.2, -0.2, sample_size).reshape(-1, 1),
        #         np.linspace(0.6**0.5, 0, sample_size).reshape(-1, 1) ** 2,
        #         np.linspace(0, 0.7, sample_size).reshape(-1, 1),
        #     ]
        # ),
    }

    for experiment_title, coefficients in experiments.items():
        logger.info(f"Running experiment: {experiment_title}")

        experiment_location = savedir / experiment_title
        os.makedirs(experiment_location, exist_ok=True)

        plot_coefficients(coefficients, experiment_location)

        num_simulations = config["num_simulations"]
        noise_variance = config["noise_variance"]
        train_payment = config["train_payment"]
        test_payment = config["test_payment"]
        regularization = config["regularization"]
        sample_size = config["sample_size"]
        forgetting_factors = config["forgetting_factors"]
        burn_in = config["burn_in"]

        interpolant_function = lambda X: np.sum(X * coefficients, axis=1, keepdims=True)
        additive_noise_function = lambda sample_size: np.random.normal(
            0, np.sqrt(noise_variance), size=(sample_size, 1)
        )
        heteroskedasticity_function = lambda X: 1

        cache_location = experiment_location / "cache"
        os.makedirs(cache_location, exist_ok=True)

        @cache(cache_location)
        def _run_experiment():
            num_forgetting_factors = len(forgetting_factors)

            experiment_results = np.zeros(
                (num_forgetting_factors, num_simulations)
            ).astype(np.object_)

            for i, forgetting_factor in tqdm(
                enumerate(config["forgetting_factors"]),
                position=0,
                desc="Forgetting Factors",
                total=num_forgetting_factors,
            ):
                market_designs = {
                    MarketDesigns.blr_nll: {
                        "task": OnlineBayesianLinearRegression,
                        "kwargs": {
                            "noise_variance": noise_variance,
                            "forgetting": forgetting_factor,
                            "regularization": regularization,
                        },
                        "market": OnlineMarket,
                        "policy": NllShapleyPolicy,
                    },
                    MarketDesigns.blr_kld_m: {
                        "task": OnlineBayesianLinearRegression,
                        "kwargs": {
                            "noise_variance": noise_variance,
                            "forgetting": forgetting_factor,
                            "regularization": regularization,
                        },
                        "market": OnlineMarket,
                        "policy": KldContributionModShapleyPolicy,
                    },
                    MarketDesigns.blr_kld_c: {
                        "task": OnlineBayesianLinearRegression,
                        "kwargs": {
                            "noise_variance": noise_variance,
                            "forgetting": forgetting_factor,
                            "regularization": regularization,
                        },
                        "market": OnlineMarket,
                        "policy": KldCfModShapleyPolicy,
                    },
                }

                iterator = tqdm(desc="Simulations", position=1, leave=False)
                with tqdm_joblib(iterator) as _:
                    experiment_results[i] = Parallel(n_jobs=-1)(
                        delayed(simulate_online_market)(
                            market_data=build_data(
                                sample_size=int(sample_size),
                                num_features=coefficients.shape[1] - 1,
                                interpolant_function=interpolant_function,
                                additive_noise_function=additive_noise_function,
                                heteroskedasticity_function=heteroskedasticity_function,
                            )(),
                            market_designs=market_designs,
                            likelihood_flattening=forgetting_factor,
                            burn_in=burn_in,
                            train_payment=train_payment,
                            test_payment=test_payment,
                        )
                        for _ in range(num_simulations)
                    )
            return parse_results(
                experiment_results,
                num_simulations,
                num_forgetting_factors,
                coefficients.shape[1] - 2,
                sample_size,
                ("train", "test"),
                ("payments", "allocations", "contributions"),
                market_designs.keys(),
            )

        parsed_results = _run_experiment()
        plot_metric_boostrap(parsed_results, experiment_location, idx=0)


if __name__ == "__main__":
    np.random.seed(42)
    main()
