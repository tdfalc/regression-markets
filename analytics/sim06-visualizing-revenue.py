from typing import Dict, Callable
from pathlib import Path
import os
from collections import defaultdict
from itertools import cycle
from functools import partial

import numpy as np
from matplotlib.lines import Line2D
from tqdm import tqdm
from joblib import delayed, Parallel
from matplotlib import pyplot as plt

from regression_markets.market.task import (
    BayesianLinearRegression,
    MaximumLikelihoodLinearRegression,
)
from regression_markets.common.utils import tqdm_joblib, cache
from regression_markets.common.log import create_logger
from analytics.helpers import (
    save_figure,
    build_input,
    bootstrap_resample,
    MarketDesigns,
    set_style,
)
from regression_markets.market.data import BatchData
from regression_markets.market.policy import (
    KldCfModShapleyPolicy,
    KldContributionModShapleyPolicy,
    NllShapleyPolicy,
)
from regression_markets.market.mechanism import BatchMarket
from tfds.plotting import use_tex, prettify

use_tex()
set_style()


def conditional_value_at_risk(
    x: np.ndarray, alpha: float, axis: int = 0
) -> np.ndarray:
    return -np.mean(
        np.partition(x, int(x.shape[axis] * alpha), axis=axis)[
            : int(x.shape[axis] * alpha)
        ],
        axis=axis,
    )


def build_data(
    train_size: int,
    test_size: float,
    num_features: int,
    interpolant_function: Callable,
    additive_noise_function: Callable,
    heteroskedasticity_function: Callable,
) -> Callable:
    sample_size = train_size + test_size
    test_frac = test_size / sample_size

    def _build_data():
        X = build_input(sample_size, num_features)
        noise = additive_noise_function(sample_size)
        y = interpolant_function(X)
        y += noise * heteroskedasticity_function(X)
        return BatchData(X[:, [0]], X[:, [1]], X[:, 2:], y, test_frac=test_frac)

    return _build_data


def simulate_batch(
    market_data: BatchData,
    market_designs: dict,
    train_payment: float,
    test_payment: float,
) -> Dict:
    def calculate_losses(task):
        X_train, X_test = market_data.X_train, market_data.X_test
        y_train, y_test = market_data.y_train, market_data.y_test
        indices = np.arange(market_data.X_train.shape[1])
        task.update_posterior(X_train, y_train, indices)
        posterior = task.get_posterior(indices)
        noise_variance = task.get_noise_variance(indices)
        train_loss = task.calculate_loss(
            X_train, y_train, posterior, noise_variance
        )
        test_loss = task.calculate_loss(
            X_test, y_test, posterior, noise_variance
        )
        return train_loss, test_loss

    results = {}
    for market_design, config in market_designs.items():
        result = {"train": {}, "test": {}}
        task = config["task"](**config["kwargs"])
        train_loss, test_loss = calculate_losses(task)
        result["train"]["loss"] = train_loss
        result["test"]["loss"] = test_loss
        market = config["market"](
            market_data,
            task,
            train_payment=train_payment,
            test_payment=test_payment,
        )
        output = market.run(config["policy"])
        for metric in ("payments", "allocations", "contributions"):
            result["train"][metric] = output["train"][metric]
            result["test"][metric] = output["test"][metric]
        results[market_design] = result

    return results


def parse_result(results: Dict) -> Dict:
    parsed_results = partial(
        defaultdict, partial(defaultdict, partial(defaultdict, list))
    )()

    for result in results:
        for market_design, market_results in result.items():
            for stage in ("train", "test"):
                for metric in market_results[stage].keys():
                    parsed_results[market_design][stage][metric].append(
                        market_results[stage][metric]
                    )

    for market_design in parsed_results.keys():
        for stage in ("train", "test"):
            for metric in parsed_results[market_design][stage].keys():
                parsed_results[market_design][stage][metric] = np.vstack(
                    parsed_results[market_design][stage][metric]
                )

    return parsed_results


def plot_results(
    results: dict,
    experiments: dict,
    market_designs: dict,
    metric: str,
    num_bootstraps: int,
    lower_quantile: float,
    upper_quantile: float,
    savedir: Path,
) -> None:
    colors = cycle(plt.get_cmap("viridis", 5).colors[1:])
    # fig, axs = plt.subplots(4, 1, figsize=(6.2, 7), sharey=True, sharex=True)
    fig, axs = plt.subplots(3, 1, figsize=(5, 5), sharey=True, sharex=True)

    markers = cycle(["o", "d", ">", "s"])

    for ls, stage in zip(["-", "--"], ("train", "test")):
        for experiment_title in experiments.keys():
            custom_lines = []
            marker = next(markers)
            for ax, market_design in zip(axs.flatten(), market_designs.keys()):

                color = next(colors)
                experiment_results = results[experiment_title]
                samples = experiment_results[market_design][stage][metric]
                bootstraps = bootstrap_resample(
                    samples, num_bootstraps, sample_size=int(len(samples) / 10)
                )
                expected_value = np.mean(bootstraps, axis=0)
                expected_shortfall = conditional_value_at_risk(
                    bootstraps, alpha=0.05, axis=0
                )

                def get_statistics(metric):
                    mean = metric.mean(axis=0)
                    confidence_interval = []
                    for bound in (lower_quantile, upper_quantile):
                        quantile = np.quantile(metric, bound, axis=0)
                        interval = np.abs(quantile - mean)
                        confidence_interval.append(interval[0:1])
                    return mean[0:1], confidence_interval

                x, x_interval = get_statistics(expected_shortfall)
                y, y_interval = get_statistics(expected_value)

                def make_errorbar(ax):
                    (_, caps, line) = ax.errorbar(
                        x,
                        y,
                        marker=marker,
                        yerr=y_interval,
                        xerr=x_interval,
                        capsize=3,
                        color=color,
                        label=market_design,
                        markerfacecolor=color,
                        markersize=6,
                        lw=1,
                        markeredgewidth=0.8,
                        markeredgecolor="k",
                    )
                    line[0].set_linestyle(ls)
                    for cap in caps:
                        cap.set_markeredgewidth(1)

                custom_lines.append(Line2D([0], [0], color=color, lw=1))
                make_errorbar(ax)

    for j, ax in enumerate(axs.flatten()):
        prettify(ax=ax, legend=False)
        if j == 1:
            ax.legend(
                custom_lines,
                [
                    market_design.value
                    for market_design in market_designs.keys()
                ],
                framealpha=0,
                ncol=2,
                loc="upper right",
                bbox_to_anchor=(1, 1.05),
            )
        ax.axvline(x=0, color="lightgray", lw=1)
        ax.axhline(y=0, color="lightgray", lw=1)

        ax.set_ylabel(r"$\mathbb{E}[r]$")
        if j == 2:
            ax.set_xlabel(r"$\mathbb{E}[r \vert r \leq Q_{\alpha} (r)]$")
        ax.set_xscale("symlog", linthresh=1e-2)
        ax.set_yscale("symlog", linthresh=1e-1)
        ax.set_xlim([-0.005, 80])
        ax.set_ylim([-5, 3])

        ax.yaxis.set_tick_params(labelbottom=True)

    # prettify(ax=ax, legend_loc="upper left", legend=False)
    save_figure(fig, savedir, f"revenue")


def main() -> None:
    logger = create_logger(__name__)
    logger.info("Running visualizing revenue analysis")

    savedir = Path(__file__).parent / "docs/sim06-visualizing-revenue"
    os.makedirs(savedir, exist_ok=True)

    set_style()

    train_size = 10
    test_size = 10
    coefficients = np.atleast_2d([0.1, -0.5, 0, 0.7]).T
    noise_variance = 1.5
    regularization = 1e-32
    num_simulations = 1000
    train_payment = 0.03
    test_payment = 0.03

    # Specified market designs
    market_designs = {
        # MarketDesigns.mle_nll: {
        #     "task": MaximumLikelihoodLinearRegression,
        #     "kwargs": {
        #         "noise_variance": noise_variance,
        #     },
        #     "market": BatchMarket,
        #     "policy": NllShapleyPolicy,
        # },
        MarketDesigns.blr_nll: {
            "task": BayesianLinearRegression,
            "kwargs": {
                "noise_variance": noise_variance,
                "regularization": regularization,
            },
            "market": BatchMarket,
            "policy": NllShapleyPolicy,
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
        MarketDesigns.blr_kld_c: {
            "task": BayesianLinearRegression,
            "kwargs": {
                "noise_variance": noise_variance,
                "regularization": regularization,
            },
            "market": BatchMarket,
            "policy": KldCfModShapleyPolicy,
        },
    }

    # Well-specified model
    interpolant_function = lambda X: X @ coefficients
    additive_noise_function = lambda sample_size: np.random.normal(
        0,
        np.sqrt(noise_variance),
        size=(sample_size, 1),
    )
    heteroskedasticity_function = lambda X: 1

    # Induced misspecifications
    misspecified_interpolant_function = lambda X: X**2 @ coefficients
    misspecified_additive_noise_function = (
        lambda sample_size: np.random.standard_t(df=2, size=(sample_size, 1))
    )
    misspecified_heteroskedasticity_function = lambda X: X[:, -2:-1] ** 2
    misspecified_heteroskedasticity_function = lambda X: 2 * X[:, -1:]

    experiments = {
        "Baseline": {
            "build_data": build_data(
                train_size=train_size,
                test_size=test_size,
                num_features=len(coefficients) - 1,
                interpolant_function=interpolant_function,
                additive_noise_function=additive_noise_function,
                heteroskedasticity_function=heteroskedasticity_function,
            ),
        },
        "+Interpolant": {
            "build_data": build_data(
                train_size=train_size,
                test_size=test_size,
                num_features=len(coefficients) - 1,
                interpolant_function=misspecified_interpolant_function,
                additive_noise_function=additive_noise_function,
                heteroskedasticity_function=heteroskedasticity_function,
            ),
        },
        "+Noise": {
            "build_data": build_data(
                train_size=train_size,
                test_size=test_size,
                num_features=len(coefficients) - 1,
                interpolant_function=misspecified_interpolant_function,
                additive_noise_function=misspecified_additive_noise_function,
                heteroskedasticity_function=heteroskedasticity_function,
            ),
        },
        "+Heteroskedasticity": {
            "build_data": build_data(
                train_size=train_size,
                test_size=test_size,
                num_features=len(coefficients) - 1,
                interpolant_function=misspecified_interpolant_function,
                additive_noise_function=misspecified_additive_noise_function,
                heteroskedasticity_function=misspecified_heteroskedasticity_function,
            ),
        },
    }

    all_results = defaultdict(dict)
    for experiment_title, experiment in experiments.items():
        logger.info(f"Running experiment: {experiment_title}")
        cache_location = savedir / "cache" / experiment_title
        os.makedirs(cache_location, exist_ok=True)
        with tqdm_joblib(
            tqdm(desc="Simulation progress", total=num_simulations)
        ) as _:
            experiment_results = cache(save_dir=cache_location)(
                lambda: Parallel(n_jobs=-1)(
                    delayed(simulate_batch)(
                        market_data=experiment["build_data"](),
                        market_designs=market_designs,
                        train_payment=train_payment,
                        test_payment=test_payment,
                    )
                    for _ in range(num_simulations)
                )
            )()

        all_results[experiment_title] = parse_result(experiment_results)

    plot_results(
        results=all_results,
        experiments=experiments,
        market_designs=market_designs,
        metric="payments",
        num_bootstraps=1000,
        lower_quantile=0.05,
        upper_quantile=0.95,
        savedir=savedir,
    )


if __name__ == "__main__":
    np.random.seed(51)
    main()
