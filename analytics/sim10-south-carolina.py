import os
from typing import Sequence, Dict
from collections import defaultdict
from pathlib import Path
import math
from joblib import Parallel, delayed

import pandas as pd
import json
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
from matplotlib import pyplot as plt
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from tqdm import tqdm
from tfds.plotting import use_tex, prettify

from regression_markets.market.data import BatchData, MarketData
from regression_markets.market.task import (
    BayesianLinearRegression,
    MaximumLikelihoodLinearRegression,
    Task,
)
from regression_markets.market.policy import NllShapleyPolicy, SemivaluePolicy
from regression_markets.market.mechanism import BatchMarket, OnlineMarket
from regression_markets.common.log import create_logger
from analytics.helpers import save_figure, set_style
from regression_markets.common.utils import cache

use_tex()
set_style()


class NllBanzhafPolicy(NllShapleyPolicy):
    def __init__(
        self,
        active_agents: np.ndarray,
        baseline_agents: np.ndarray,
        polynomial_degree: int,
        regression_task: Task,
        observational: bool = True,
    ):
        super().__init__(
            active_agents=active_agents,
            baseline_agents=baseline_agents,
            polynomial_degree=polynomial_degree,
            regression_task=regression_task,
            observational=observational,
        )
        self.observational = observational

    def _contribution_weight(self, *_) -> float:
        return 1 / (2 ** (len(self.active_agents) - 1))


class NllWeightedShapleyPolicy(NllShapleyPolicy):
    def __init__(
        self,
        active_agents: np.ndarray,
        baseline_agents: np.ndarray,
        polynomial_degree: int,
        regression_task: Task,
        observational: bool = True,
    ):
        super().__init__(
            active_agents=active_agents,
            baseline_agents=baseline_agents,
            polynomial_degree=polynomial_degree,
            regression_task=regression_task,
            observational=observational,
        )
        self.median_coalition_size = 0.5 * (self.grand_coalition_size - 1)
        self.floor = math.floor(self.median_coalition_size)
        self.ceil = math.ceil(self.median_coalition_size)

    def _contribution_weight(self, coalition_size: int) -> float:
        if coalition_size >= self.floor:
            return super()._contribution_weight(coalition_size)
        return (
            super()._contribution_weight(coalition_size)
            * self.ceil
            * self.floor
            / math.factorial(coalition_size)
            / math.factorial(self.grand_coalition_size - coalition_size - 1)
        )


class NllPenalizedShapleyPolicy(NllShapleyPolicy):
    def __init__(
        self,
        active_agents: np.ndarray,
        baseline_agents: np.ndarray,
        polynomial_degree: int,
        regression_task: Task,
        observational: bool = True,
        alpha: float = 0.35,  # np.log(1.5),
    ):
        super().__init__(
            active_agents=active_agents,
            baseline_agents=baseline_agents,
            polynomial_degree=polynomial_degree,
            regression_task=regression_task,
            observational=observational,
        )
        self.alpha = alpha

    def _weighted_avg_contributions(
        self, X: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        contributions = super()._weighted_avg_contributions(X, y)

        def _calculate_penalties(X: np.ndarray):
            X_normalized = X / np.linalg.norm(X, axis=1, keepdims=True)
            similarity_matrix = np.dot(X_normalized, X_normalized.T)
            np.fill_diagonal(similarity_matrix, 0)
            support_agents = list(self.active_agents)
            return similarity_matrix.sum(axis=1)[support_agents]

        penalties = _calculate_penalties(X.T)
        return contributions * np.exp(-self.alpha * penalties)


def noise_variance_mle(data: BatchData) -> float:
    task = MaximumLikelihoodLinearRegression()
    indices = np.arange(data.X.shape[1])
    task.update_posterior(data.X, data.y, indices)
    return task.get_noise_variance(indices)


def parse_raw_data(fname: Path) -> pd.DataFrame:
    df = pd.read_csv(fname, header=None, index_col=0)
    df.index = pd.to_datetime(df.index, format="%d/%m/%Y %H.%M")
    df = df.tz_localize("utc")
    df.columns = [f"a{i}" for i in df.columns]
    return df


def build_market_data(
    df, test_frac, central_agent, num_replications, agents, num_samples=None
) -> BatchData:
    df = df.resample("1h").mean()
    df = df.loc[:, agents].copy()

    for agent in agents:
        df[f"{agent}_lag1"] = df[f"{agent}"].shift(1)
        if agent != central_agent:
            df = df.drop(f"{agent}", axis=1)

    # df /= df.max(axis=0)
    df = df.dropna()

    if num_samples is not None:
        idx = np.random.randint(0, len(df) - num_samples - 1)
        df = df.iloc[idx : idx + num_samples]

    target_signal = df.pop(f"{central_agent}").to_numpy().reshape(-1, 1)
    central_agent_cols = [f"{central_agent}_lag1"]
    central_agent_features = df.loc[:, central_agent_cols]
    support_agent_features = df.loc[
        :, [c for c in df.columns if c not in central_agent_cols]
    ]
    support_agent_features = support_agent_features.to_numpy()

    if num_replications > 0:
        support_agent_features = np.hstack(
            [
                (
                    support_agent_features[:, 0]
                    + np.random.normal(
                        0, 0.12, size=len(support_agent_features)
                    )
                ).reshape(-1, 1)
                for _ in range(num_replications)
            ]
            + [support_agent_features]
        )

    return BatchData(
        dummy_feature=np.ones((len(target_signal), 1)),
        central_agent_features=central_agent_features.to_numpy(),
        support_agent_features=support_agent_features,
        target_signal=target_signal,
        polynomial_degree=1,
        test_frac=test_frac,
    )


def plot_raw_data(raw_data: pd.DataFrame, savedir: Path) -> None:
    fig, ax = plt.subplots(dpi=300, figsize=(6, 4))
    for wf, data in raw_data.groupby(pd.Grouper(freq="SME")).mean().items():
        ax.plot(data, label=wf)
    ax.legend(ncol=5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized Output")
    ax.set_title("Bi-weekly Average Wind Farm Output")
    prettify(ax=ax, legend=False)
    save_figure(fig, savedir, "raw_data")


def plot_map(
    shapefile: Path,
    raw_data: pd.DataFrame,
    locations: Dict,
    savedir: Path,
) -> None:
    gdf_shapefile = gpd.read_file(shapefile)
    geometry = [
        Point(lon, lat)
        for lon, lat in zip(locations["longitude"], locations["latitude"])
    ]
    gdf_points = gpd.GeoDataFrame(
        locations, geometry=geometry, crs=gdf_shapefile.crs
    )
    rd = raw_data.copy()
    gdf_points["correlation"] = rd.corr()["a1"].values

    gdf_points = gdf_points.sort_values(by="correlation")

    normalized_correlation = 0.1 + (
        gdf_points["correlation"] - gdf_points["correlation"].min()
    ) / (gdf_points["correlation"].max() - gdf_points["correlation"].min())
    marker_size = [value * 500 for value in normalized_correlation]

    fig, ax = plt.subplots(dpi=600, figsize=(3.6, 3.2))
    fig, ax = plt.subplots(figsize=(6.5, 6), dpi=300)
    gdf_shapefile.plot(ax=ax, color="k", edgecolor="w", alpha=0.2, lw=1)
    gdf_points.plot(
        ax=ax, c="None", markersize=marker_size, edgecolor="r", lw=2
    )

    for x, y, label in zip(
        gdf_points.geometry.x, gdf_points.geometry.y, gdf_points["agent"]
    ):
        if label == "a1":
            xytext = (10, 35)

        if label == "a2":
            xytext = (30, 20)

        if label == "a3":
            xytext = (35, 40)

        if label == "a4":
            xytext = (70, 15)

        if label == "a5":
            xytext = (-30, -50)

        if label == "a6":
            xytext = (-40, -30)

        if label == "a7":
            xytext = (-55, -25)

        if label == "a8":
            xytext = (-55, -45)

        if label == "a9":
            xytext = (35, -25)

        ax.annotate(
            label,
            xy=(x, y),
            xytext=xytext,
            xycoords="data",
            textcoords="offset points",
            fontsize=10,
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
        )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    prettify(ax=ax, legend=False)
    save_figure(fig, savedir, "map")


def _add_replicate_revenue(
    values: np.ndarray, max_replications: int
) -> np.ndarray:
    values = values.copy()
    total = values[: max_replications + 1].sum()
    values = values[max_replications:]
    values[0] = total
    return values


def plot_results(results: Dict, max_replications: int, savedir: Path) -> None:
    bar_width = 0.2
    metric = "allocations"
    cmap = plt.get_cmap("viridis", 6)
    colors = cmap.colors[[2, -2]]

    for policy in ("Shapley-Obs", "Shapley-Int"):

        fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
        # fig, ax = plt.subplots(figsize=(6.5, 3), dpi=300)

        for stage, color in zip(("train", "test"), colors):
            allocations_before = (
                results[policy][0][stage][metric].flatten() * 100
            )
            allocations_after = (
                results[policy][max_replications][stage][metric].flatten() * 100
            )

            # Combine revenue of replicated features
            allocations_after_agg = _add_replicate_revenue(
                allocations_after, max_replications
            )

            # First we add total allocation bars
            hatch = "//"
            positions = np.arange(len(allocations_before))
            ax.bar(
                (
                    (positions - 1.5 * bar_width)[0]
                    if stage == "train"
                    else (positions - 0.5 * bar_width)[0]
                ),
                allocations_before[0],
                bar_width,
                color=color,
                edgecolor="k",
                hatch=hatch,
            )
            ax.bar(
                (
                    (positions - 1.5 * bar_width)[1:]
                    if stage == "train"
                    else (positions - 0.5 * bar_width)[1:]
                ),
                allocations_before[1:],
                bar_width,
                color=color,
                edgecolor="k",
                hatch=hatch,
            )

            hatch = ".."
            ax.bar(
                (
                    (positions + 0.5 * bar_width)[1:]
                    if stage == "train"
                    else (positions + 1.5 * bar_width)[1:]
                ),
                allocations_after_agg[1:],
                bar_width,
                color=color,
                edgecolor="k",
                hatch=hatch,
            )

            # Now we stack bars for replicates
            if policy == "Shapley-Obs":
                bottom = 0
                for i, replicate in enumerate(
                    list(range(max_replications + 1))[::-1]
                ):
                    heights = np.zeros(len(allocations_before))
                    heights[0] = allocations_after[replicate]

                    ax.bar(
                        (
                            (positions + 0.5 * bar_width)[0]
                            if stage == "train"
                            else (positions + 1.5 * bar_width)[0]
                        ),
                        heights[0],
                        bar_width,
                        color=color,
                        edgecolor="k",
                        bottom=bottom,
                        hatch=hatch,
                        alpha=0.7 if i > 0 else 1,
                        # lw=2,
                    )
                    if i > 0:
                        ax.bar(
                            (
                                (positions + 0.5 * bar_width)[0]
                                if stage == "train"
                                else (positions + 1.5 * bar_width)[0]
                            ),
                            heights[0],
                            bar_width,
                            color="None",
                            edgecolor="red",
                            bottom=bottom,
                            alpha=1,
                            lw=1.5,
                            zorder=10,
                        )
                    bottom += heights[0]

                legend_elements = [
                    Patch(facecolor=colors[0], edgecolor="k", label="Train"),
                    Patch(
                        facecolor=colors[1],
                        edgecolor="k",
                        label="Test",
                    ),
                ]
                ax.legend(
                    handles=legend_elements, fancybox=False, edgecolor="white"
                )

            else:
                ax.bar(
                    (
                        (positions + 0.5 * bar_width)[0]
                        if stage == "train"
                        else (positions + 1.5 * bar_width)[0]
                    ),
                    allocations_after_agg[0],
                    bar_width,
                    color=color,
                    edgecolor="k",
                    hatch=hatch,
                )

        ax.set_xlabel("Support Agent")
        ax.set_ylabel("Revenue Allocation ($\%$)")
        ax.set_ylim(top=33)
        ax.set_xticks(np.arange(len(allocations_before)))
        ax.set_xticklabels(
            ["$a_4$", "$a_5$", "$a_6$", "$a_7$", "$a_8$", "$a_9$"]
        )
        prettify(ax=ax)

        fig.tight_layout()
        save_figure(
            fig,
            savedir,
            f"allocations_{policy.lower()}".replace("-", "_").lower(),
        )


def plot_policy_comparison(
    results: Dict, policies: Sequence, max_replications: int, savedir: Path
):
    stage = "train"
    metric = "allocations"

    # fig, ax = plt.subplots(figsize=(6, 3), dpi=300)
    # fig, ax = plt.subplots(figsize=(6.5, 3), dpi=300)
    fig, ax = plt.subplots(figsize=(5, 3), dpi=300)

    labels = [
        "Interventional",
        "Observational",
        "Banzahf Value",
        "Robust-Shapley",
    ]

    policy_allications = {}
    markers = ["*", "s", "o", "^"]
    colors = plt.get_cmap("viridis", 4).colors
    for i, policy in enumerate(policies):
        all_allocations = [results[policy][0][stage][metric].flatten() * 100]

        for num_replications in range(1, max_replications + 1):
            allocations = (
                results[policy][num_replications][stage][metric].flatten() * 100
            )
            allocations_agg = _add_replicate_revenue(
                allocations, num_replications
            )
            all_allocations.append(allocations_agg)

        policy_allications[policy] = np.row_stack(all_allocations)[:, 0]

        ax.plot(
            policy_allications[policy],
            marker=markers[i],
            # label=policy,
            label=labels[i],
            markerfacecolor="None",
            color=colors[i],
            markersize=7,
        )
        ax.set_ylabel("Revenue Allocation ($\%$)")
        ax.set_xlabel("Number of Replications")

    prettify(ax=ax, legend_loc="upper left", legend=False)
    leg = ax.legend(
        facecolor="#eeeeee",
        edgecolor="#ffffff",
        framealpha=0.8,
        loc="upper left",
        labelspacing=0.25,
    )
    leg.get_frame().set_linewidth(0)
    # ax.get_frame().set_linewidth(0)
    fig.tight_layout()
    save_figure(fig, savedir, f"policy_comparison")


if __name__ == "__main__":
    logger = create_logger(__name__)
    logger.info("Running south carolina analysis")

    use_tex()

    savedir = Path(__file__).parent / "docs/sim10-south-carolina"
    os.makedirs(savedir, exist_ok=True)

    fname = savedir / "sc_wind_power.csv"
    raw_data = parse_raw_data(fname)

    with open(savedir / "locations.json", "r") as f:
        locations = json.load(f)

    plot_raw_data(raw_data, savedir)
    plot_map(savedir / "sc_shapefile", raw_data, locations, savedir)

    test_frac = 0.1
    train_payment = 2
    test_payment = 2
    num_samples = None
    central_agent = "a1"
    regularization = 1e-32
    max_replications = 4  # 5  # 6
    agents = ["a1", "a4", "a5", "a6", "a7", "a8", "a9"]
    experiments = {
        "Shapley-Int": {
            "policy": NllShapleyPolicy,
            "observational": False,
        },
        "Shapley-Obs": {
            "policy": NllShapleyPolicy,
            "observational": True,
        },
        "Banzhaf-Obs": {
            "policy": NllBanzhafPolicy,
            "observational": True,
        },
        "Shapley-Obs-Penalized": {
            "policy": NllPenalizedShapleyPolicy,
            "observational": True,
        },
        # "Shapley-Obs-Weighted": {
        #     "policy": NllWeightedShapleyPolicy,
        #     "observational": True,
        # },
    }

    cache_location = savedir / "cache"
    os.makedirs(cache_location, exist_ok=True)

    @cache(cache_location)
    def _run_experiments():
        results = defaultdict(dict)

        def _run_experiment(experiment):
            policy = experiment["policy"]
            observational = experiment["observational"]
            experiment_results = dict()
            for num_replications in tqdm(range(0, max_replications + 1)):
                raw_data = parse_raw_data(fname)
                market_data = build_market_data(
                    raw_data,
                    test_frac,
                    central_agent=central_agent,
                    agents=agents,
                    num_replications=num_replications,
                    num_samples=num_samples,
                )

                task = BayesianLinearRegression(
                    regularization=regularization,
                    noise_variance=noise_variance_mle(market_data),
                )
                market = BatchMarket(
                    market_data,
                    regression_task=task,
                    observational=observational,
                    train_payment=train_payment,
                    test_payment=test_payment,
                )

                experiment_results[num_replications] = market.run(policy)
            return experiment_results

        results = Parallel(n_jobs=-1)(
            delayed(_run_experiment)(experiment)
            for experiment in experiments.values()
        )

        return {name: results[i] for i, name in enumerate(experiments.keys())}

    results = _run_experiments()

    plot_results(results, max_replications=4, savedir=savedir)
    plot_policy_comparison(
        results,
        experiments.keys(),
        max_replications,
        savedir,
    )
