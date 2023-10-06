import os
from typing import Sequence, Dict
from collections import defaultdict
from pathlib import Path

import pandas as pd
import json
from matplotlib.patches import Patch
from matplotlib import pyplot as plt
import numpy as np
import geopandas as gpd
from shapely.geometry import Point

from market.data import BatchData
from market.task import (
    BayesianLinearRegression,
    MaximumLikelihoodLinearRegression,
)
from market.policy import NllShapleyPolicy
from market.mechanism import BatchMarket
from common.log import create_logger
from analytics.helpers import save_figure, ggplot_colors
from common.utils import cache


def noise_variance_mle(data: BatchData):
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
):
    df = df.resample("1H").mean()
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


def plot_raw_data(raw_data, savedir: Path):
    fig, ax = plt.subplots(dpi=300, figsize=(6, 4))
    for wf, data in raw_data.groupby(pd.Grouper(freq="SM")).mean().iteritems():
        ax.plot(data, label=wf)
    ax.legend(ncol=5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized Output")
    ax.set_title("Bi-weekly Average Wind Farm Output")
    save_figure(fig, savedir, "raw_data")


def plot_map(
    shapefile: Path,
    raw_data: pd.DataFrame,
    locations: Dict,
    savedir: Path,
):
    gdf_shapefile = gpd.read_file(shapefile)
    geometry = [
        Point(lon, lat)
        for lon, lat in zip(locations["longitude"], locations["latitude"])
    ]
    gdf_points = gpd.GeoDataFrame(
        locations, geometry=geometry, crs=gdf_shapefile.crs
    )
    gdf_points["correlation"] = raw_data.corr()["a1"].values
    gdf_points["colors"] = ggplot_colors()[:9]
    gdf_points = gdf_points.sort_values(by="correlation")

    normalized_correlation = 0.1 + (
        gdf_points["correlation"] - gdf_points["correlation"].min()
    ) / (gdf_points["correlation"].max() - gdf_points["correlation"].min())
    marker_size = [value * 500 for value in normalized_correlation]

    fig, ax = plt.subplots(dpi=600, figsize=(3.6, 3))
    gdf_shapefile.plot(ax=ax, color="whitesmoke", edgecolor="gray", alpha=1)
    gdf_points.plot(
        ax=ax, c=gdf_points["colors"], markersize=marker_size, edgecolor="k"
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
    save_figure(fig, savedir, "map")


def plot_acf(agents: Sequence, savedir):
    def acf(x, length=50):
        return np.array(
            [1] + [np.corrcoef(x[:-i], x[i:])[0, 1] for i in range(1, length)]
        )

    fig, ax = plt.subplots(dpi=600, figsize=(3.6, 3))
    colors = ggplot_colors()[:9]

    for i, wf in enumerate(agents):
        ax.plot(acf(raw_data[wf], length=50), c=colors[i], label=wf)

    ax.set_xlabel("Lag")
    ax.set_ylabel("Correlation")
    ax.legend(ncol=3)

    save_figure(fig, savedir, "acf")


def plot_results(results: Dict, max_replications: int, savedir: Path):
    bar_width = 0.2
    metric = "allocations"
    hatches = ("//", "..")
    colors = ggplot_colors()[:2]

    for market_type in ("obs", "int"):
        fig, ax = plt.subplots(figsize=(3.6, 3), dpi=300)

        for stage, hatch, color in zip(("train", "test"), hatches, colors):

            def add_replicate_revenue(values):
                values = values.copy()
                total = values[: max_replications + 1].sum()
                values = values[max_replications:]
                values[0] = total
                return values

            allocations_before = (
                results[market_type][0][stage][metric].flatten() * 100
            )
            allocations_after = (
                results[market_type][1][stage][metric].flatten() * 100
            )

            # Combine revenue of replicated features
            allocations_after_agg = add_replicate_revenue(allocations_after)

            # First we add total allocation bars
            positions = np.arange(len(allocations_before))
            ax.bar(
                (positions - 1.5 * bar_width)[0]
                if stage == "train"
                else (positions - 0.5 * bar_width)[0],
                allocations_before[0],
                bar_width,
                color=color,
                edgecolor="k",
                hatch=hatch,
            )
            ax.bar(
                (positions + 0.5 * bar_width)[0]
                if stage == "train"
                else (positions + 1.5 * bar_width)[0],
                allocations_after_agg[0],
                bar_width,
                color=color,
                edgecolor="k",
                hatch=hatch,
            )

            ax.bar(
                (positions - 1.5 * bar_width)[1:]
                if stage == "train"
                else (positions - 0.5 * bar_width)[1:],
                allocations_before[1:],
                bar_width,
                color=color,
                edgecolor="k",
                hatch=hatch,
            )

            ax.bar(
                (positions + 0.5 * bar_width)[1:]
                if stage == "train"
                else (positions + 1.5 * bar_width)[1:],
                allocations_after_agg[1:],
                bar_width,
                color=color,
                edgecolor="k",
                hatch=hatch,
            )

            # Now we stack bars for replicates
            if market_type == "obs":
                bottom = 0
                for replicate in list(range(max_replications + 1))[::-1]:
                    heights = np.zeros(len(allocations_before))
                    heights[0] = allocations_after[replicate]
                    ax.bar(
                        (positions + 0.5 * bar_width)[0]
                        if stage == "train"
                        else (positions + 1.5 * bar_width)[0],
                        heights[0],
                        bar_width,
                        color=color,
                        edgecolor="k",
                        bottom=bottom,
                        hatch=hatch,
                    )
                    bottom += heights[0]

        legend_elements = [
            Patch(facecolor=colors[0], edgecolor="k", label="w/. Honest"),
            Patch(facecolor=colors[1], edgecolor="k", label="w/. Malicious"),
        ]
        ax.legend(handles=legend_elements, fancybox=False, edgecolor="white")

        ax.set_xlabel("Support Agent")
        ax.set_ylabel("Revenue Allocation (%)")
        ax.set_ylim(top=33)
        ax.set_xticks(np.arange(len(allocations_before)))
        ax.set_xticklabels(
            ["$a_4$", "$a_5$", "$a_6$", "$a_7$", "$a_8$", "$a_9$"]
        )

        fig.tight_layout()
        save_figure(fig, savedir, f"allocations_{market_type}")


if __name__ == "__main__":
    logger = create_logger(__name__)
    logger.info("Running south carolina analysis")

    savedir = Path(__file__).parent / "docs/sim12-south-carolina"
    os.makedirs(savedir, exist_ok=True)

    fname = savedir / "sc_wind_power.csv"
    raw_data = parse_raw_data(fname)

    with open(savedir / "locations.json", "r") as f:
        locations = json.load(f)

    plot_raw_data(raw_data, savedir)
    plot_map(savedir / "sc_shapefile", raw_data, locations, savedir)
    plot_acf(locations["agent"], savedir)

    test_frac = 0.1
    train_payment = 2
    test_payment = 2
    num_samples = None
    central_agent = "a1"
    regularization = 1e-32
    max_replications = 4
    agents = ["a1", "a4", "a5", "a6", "a7", "a8", "a9"]

    cache_location = savedir / "cache"
    os.makedirs(cache_location, exist_ok=True)

    @cache(cache_location)
    def _run_experiemnt():
        results = defaultdict(list)
        for num_replications in (0, max_replications):
            raw_data = parse_raw_data(fname)
            market_data = build_market_data(
                raw_data,
                test_frac,
                central_agent=central_agent,
                agents=agents,
                num_replications=num_replications,
                num_samples=num_samples,
            )

            for observational in (True, False):
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
                results["obs" if observational else "int"].append(
                    market.run(NllShapleyPolicy)
                )
        return results

    results = _run_experiemnt()

    plot_results(results, max_replications, savedir)
