from typing import Sequence, Dict
from joblib import delayed, Parallel
from pathlib import Path
import os
from collections import defaultdict
from itertools import cycle
from functools import partial

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

from common.log import create_logger
from common.utils import tqdm_joblib, cache
from analytics.helpers import (
    save_figure,
    nested_defaultdict,
    build_data,
    MarketDesigns,
    set_style,
)
from market.data import BatchData
from market.mechanism import BatchMarket
from market.policy import (
    NllShapleyPolicy,
    KldContributionModShapleyPolicy,
    KldCfModShapleyPolicy,
)
from market.task import (
    BayesianLinearRegression,
    MaximumLikelihoodLinearRegression,
)


def simulate_batch(
    market_data: BatchData,
    market_designs: dict,
    train_payment: float,
    test_payment: float,
) -> Dict:
    results = {}
    for market_design, config in market_designs.items():
        task = config["task"](**config["kwargs"])
        market = config["market"](
            market_data,
            task,
            train_payment=train_payment,
            test_payment=test_payment,
        )
        output = market.run(config["policy"])
        results[market_design] = output
    return results


def parse_results(
    results: Sequence[dict],
    num_simulations: int,
    num_agent_coefficients: int,
    num_train_sizes: int,
    num_sellers: int,
    stages: Sequence[str],
    metrics: Sequence[str],
    market_designs: Sequence[str],
) -> Dict:
    parsed_results = nested_defaultdict(
        4,
        lambda: np.zeros((num_agent_coefficients, num_train_sizes, num_sellers)),
    )

    for i in range(num_simulations):
        for j in range(num_agent_coefficients):
            for k in range(num_train_sizes):
                for market_design in market_designs:
                    for stage in stages:
                        for metric in metrics:
                            parsed_results[i][market_design][stage][metric][j, k] = (
                                results[j, k][i][market_design][stage][metric]
                            )

    stacked_results = partial(defaultdict, partial(defaultdict, dict))()
    for market_design in market_designs:
        for stage in ("train", "test"):
            for metric in metrics:
                stacked_results[market_design][stage][metric] = np.stack(
                    [
                        np.stack(result[market_design][stage][metric], axis=2)
                        for result in parsed_results.values()
                    ],
                    axis=3,
                )

    return stacked_results


def plot_shapley_convergence(
    results: dict,
    market_designs: Sequence,
    train_sizes: np.ndarray,
    agent_coefficient_index: int,
    savedir: Path,
) -> None:
    fig1, ax1 = plt.subplots(1, figsize=(6.2, 2.6), sharex=True, sharey=False)
    fig2, ax2 = plt.subplots(1, figsize=(6.2, 2.6), sharex=True, sharey=False)

    colors = cycle(["magenta", "blue", "darkorange", "limegreen"])
    line_styles = cycle(["-", "--"])

    custom_lines = []
    for market_design in market_designs:
        market_results = results[market_design]
        contributions = market_results["train"]["contributions"]
        allocations = market_results["train"]["allocations"]
        color = next(colors)
        for seller in (0, 1):
            ax1.plot(
                train_sizes,
                contributions.mean(axis=3)[:, seller, agent_coefficient_index],
                ls=next(line_styles),
                markerfacecolor="None",
                lw=1,
                color=color,
                markersize=6,
            )

        if market_design == MarketDesigns.blr_kld_m:

            ax2.plot(
                train_sizes,
                allocations.sum(axis=1).mean(axis=2)[:, agent_coefficient_index],
                ls="solid",
                markerfacecolor="None",
                lw=1,
                color=color,
                markersize=6,
            )

        custom_lines.append(Line2D([0], [0], color=color, lw=1))

    for i, ax in enumerate((ax1, ax2)):
        ax.set_xlabel("Sample Size")
        ax.set_xscale("log")
        ax.set_xticks([10, 100, 1000, 10000])
        ax.set_ylabel("Shapley Value" if i == 0 else "Total Allocation")
        if i == 0:
            ax.legend(
                custom_lines,
                [market_design.value for market_design in market_designs],
                framealpha=0,
                ncol=2,
                loc="upper right",
                bbox_to_anchor=(1, 0.92),
            )
            ax.set_ylim([0.35, 0.85])
        else:
            ax.set_ylim([0.965, 1.003])
            ax.legend(
                [custom_lines[2]],
                [
                    market_design.value
                    for market_design in market_designs
                    if market_design == MarketDesigns.blr_kld_m
                ],
                framealpha=0,
                ncol=2,
                loc="upper right",
                bbox_to_anchor=(1, 0.92),
            )

    save_figure(fig1, savedir, f"shapley_convergence", tight=True)
    save_figure(fig2, savedir, f"allocation_convergence", tight=True)


def main() -> None:
    logger = create_logger(__name__)
    logger.info("Running sensitivity analysis")

    savedir = Path(__file__).parent / "docs/sim06-sensitivity-analysis"

    set_style()

    experiments = {
        "case0": {  # Shapley convergence
            "noise_variance": 0.5,
            "regularization": 1e-32,
            "num_simulations": 1000,
            "test_payment": 0.01,
            "test_size": 1000,
            "train_sizes": np.geomspace(10, 10000, 4),
            "agent_coefficients": np.array([0.7]),
            "coefficients_function": lambda c: np.atleast_2d([-0.1, 0.8, c, -0.9]).T,
        }
    }

    for experiment_title, config in experiments.items():
        experiment_location = savedir / experiment_title
        os.makedirs(experiment_location, exist_ok=True)

        noise_variance = config["noise_variance"]
        regularization = config["regularization"]
        agent_coefficients = config["agent_coefficients"]
        train_sizes = config["train_sizes"]
        coefficients_function = config["coefficients_function"]
        test_size = config["test_size"]
        num_simulations = config["num_simulations"]
        test_payment = config["test_payment"]

        cache_location = experiment_location / "cache"
        os.makedirs(cache_location, exist_ok=True)

        num_agent_coefficients = len(agent_coefficients)
        num_train_sizes = len(train_sizes)

        logger.info(f"Running experiment: {experiment_title}")

        @cache(cache_location)
        def _run_experiment():
            experiment_results = np.zeros(
                (num_agent_coefficients, num_train_sizes, num_simulations)
            ).astype(np.object_)

            for i, agent_coefficient in tqdm(
                enumerate(agent_coefficients),
                position=0,
                desc="Agent Coefficients",
                total=num_agent_coefficients,
            ):
                for j, train_size in tqdm(
                    enumerate(train_sizes),
                    position=1,
                    desc="Sample Sizes",
                    leave=False,
                    total=num_train_sizes,
                ):
                    train_size = int(train_size)
                    train_payment = test_payment * (test_size / train_size)

                    coefficients = coefficients_function(agent_coefficient)

                    interpolant_function = lambda X: X @ coefficients
                    additive_noise_function = lambda sample_size: np.random.normal(
                        0, np.sqrt(noise_variance), size=(sample_size, 1)
                    )

                    heteroskedasticity_function = lambda X: 1

                    market_designs = {
                        MarketDesigns.mle_nll: {
                            "task": MaximumLikelihoodLinearRegression,
                            "kwargs": {
                                "noise_variance": noise_variance,
                            },
                            "market": BatchMarket,
                            "policy": NllShapleyPolicy,
                        },
                        MarketDesigns.blr_nll: {
                            "task": BayesianLinearRegression,
                            "kwargs": {
                                "noise_variance": noise_variance,
                                "regularization": regularization,
                            },
                            "market": BatchMarket,
                            "policy": NllShapleyPolicy,
                        },
                        MarketDesigns.blr_kld_c: {
                            "task": BayesianLinearRegression,
                            "kwargs": {
                                "noise_variance": noise_variance,
                                "regularization": regularization,
                            },
                            "market": BatchMarket,
                            "policy": KldCfModShapleyPolicy,
                        },
                        MarketDesigns.blr_kld_m: {
                            "task": BayesianLinearRegression,
                            "kwargs": {
                                "noise_variance": noise_variance,
                                "regularization": regularization,
                            },
                            "market": BatchMarket,
                            "policy": KldContributionModShapleyPolicy,
                        },
                    }

                    with tqdm_joblib(
                        tqdm(
                            desc="Simulations",
                            total=num_simulations,
                            position=2,
                            leave=False,
                        )
                    ) as _:
                        experiment_results[i, j] = Parallel(n_jobs=-1)(
                            delayed(simulate_batch)(
                                market_data=build_data(
                                    train_size=train_size,
                                    test_size=test_size,
                                    num_features=len(coefficients) - 1,
                                    interpolant_function=interpolant_function,
                                    additive_noise_function=additive_noise_function,
                                    heteroskedasticity_function=heteroskedasticity_function,
                                )(),
                                market_designs=market_designs,
                                train_payment=train_payment,
                                test_payment=test_payment,
                            )
                            for _ in range(num_simulations)
                        )

            return parse_results(
                experiment_results,
                num_simulations,
                num_agent_coefficients,
                num_train_sizes,
                len(coefficients) - 2,
                ("train", "test"),
                ("payments", "allocations", "contributions"),
                market_designs.keys(),
            )

        parsed_results = _run_experiment()

        plot_shapley_convergence(
            results=parsed_results,
            market_designs=MarketDesigns,
            train_sizes=train_sizes,
            agent_coefficient_index=0,
            savedir=experiment_location,
        )


if __name__ == "__main__":
    np.random.seed(42)
    main()
