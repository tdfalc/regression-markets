from pathlib import Path
import os
from typing import Sequence, Dict
from collections import defaultdict
from datetime import datetime as dt

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pytz
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
from scipy.ndimage import gaussian_filter
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

from common.log import create_logger
from common.utils import cache
from market.task import (
    MaximumLikelihoodLinearRegression,
    OnlineBayesianLinearRegression,
)
from market.data import MarketData, BatchData
from market.mechanism import OnlineMarket
from analytics.helpers import save_figure, classic_colors
from market.policy import NllShapleyPolicy


def process_raw_data(
    df: pd.DataFrame,
    start: dt,
    end: dt,
    target_code: str,
    support_agent_codes: Sequence,
    max_central_agent_lags: int,
    max_support_agent_lags: int,
):
    df = df.loc[:, [c for c in df.columns if "00" in c or c == "Time"]]
    df.loc[:, "Time"] = pd.to_datetime(df.loc[:, "Time"])
    df = df.rename(
        columns={"Time": "time"}
        | {c: c.replace("00", "") for c in df.columns if "00" in c}
    )
    df = df.set_index("time")

    # Filter data to specified date range
    df = df.tz_localize("utc")
    df = df[(df.index >= start) & (df.index <= end)]

    def add_lags(df, column, max_lags, remove_original):
        lags = [
            df[column].shift(lag).rename(f"{column}{lag}")
            for lag in range(1, max_lags + 1)
        ]
        df = pd.concat([df] + lags, axis=1)
        if remove_original:
            return df.drop(column, axis=1)
        return df

    # Add lags for all columns in the DataFrame
    for column in df.columns:
        max_lags = (
            max_central_agent_lags
            if column == target_code
            else max_support_agent_lags
        )
        df = add_lags(
            df, column, max_lags=max_lags, remove_original=column != target_code
        )

    # Drop rows with NaN values
    df = df.dropna()

    # Extract support agents features and target signal
    support_agent_features = df[
        df.columns[df.columns.str.contains("|".join(support_agent_codes))]
    ]
    target_signal = df.pop(target_code)

    # Extract central_agent features
    central_agent_features = df.filter(regex=target_code)

    return support_agent_features, central_agent_features, target_signal


def noise_variance_mle(data: BatchData):
    task = MaximumLikelihoodLinearRegression()
    indices = np.arange(data.X.shape[1])
    task.update_posterior(data.X, data.y, indices)
    return task.get_noise_variance(indices)


def plot_irradiance(
    target_signals: Sequence,
    target_codes: Sequence,
    color_map: Dict,
    savedir: Path,
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 2.5))
    for i, target_code in enumerate(target_codes):
        target_signal = target_signals[i]
        ax1.plot(
            target_signal.index,
            gaussian_filter(target_signals[i], sigma=1000),
            label=target_code,
            color=color_map[i],
        )
        ax1.set_ylabel("Irradiance $(\mathrm{kW}/\mathrm{m}^2)$")
        ax1.set_xlabel("Date")

        hourly = target_signal.groupby(target_signal.index.hour).mean()
        ax2.plot(
            hourly.index,
            hourly,
            label=target_code,
            color=color_map[i],
        )
        ax2.set_ylabel("Irradiance $(\mathrm{kW}/\mathrm{m}^2)$")
        ax2.set_xlabel("Hour")

        date_format = mdates.DateFormatter("%m/%y")
        ax1.xaxis.set_major_formatter(date_format)
        ax1.legend(ncol=2)
        ax1.xaxis.set_major_locator(MaxNLocator(5))

        date_format = mdates.DateFormatter("%m/%y")
        ax2.xaxis.set_major_locator(MaxNLocator(7))
        ax2.set_xlabel("Hour")

    save_figure(fig, savedir, "irradiance")


def plot_results(
    results: Dict,
    index: Sequence,
    target_codes: Sequence,
    color_map: Dict,
    savedir: Path,
):
    for two_target_codes in zip(target_codes[::2], target_codes[1::2]):
        fig, axs = plt.subplots(1, 2, figsize=(6, 2.5), sharey=True)
        for ax, target_code in zip(axs.flatten(), two_target_codes):
            output = results[target_code][str(NllShapleyPolicy)]
            ax.yaxis.set_tick_params(labelbottom=True)
            ax.set_ylabel("Revenue (EUR)")
            ax.set_xlabel(r"Time")

            target_code_index = target_codes.index(target_code)
            color_idx = 0
            for i in range(output["test"]["payments"].shape[1]):
                if color_idx == target_code_index:
                    color_idx += 1

                ax.plot(
                    index,
                    gaussian_filter(
                        output["train"]["payments"][:, i]
                        + output["test"]["payments"][:, i],
                        sigma=1000,
                    ),
                    color=color_map[color_idx],
                    label=target_codes[color_idx],
                )
                color_idx += 1
            print(target_code, ax.get_ylim())

            date_format = mdates.DateFormatter("%m/%y")
            ax.xaxis.set_major_formatter(date_format)

            from matplotlib.ticker import MaxNLocator

            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.set_ylim([-0.05, 1.1])

        save_figure(fig, savedir, "_".join(two_target_codes))


def main():
    logger = create_logger(__name__)
    logger.info("Running solar (PECD) analysis")

    savedir = Path(__file__).parent / "docs/sim08-solar-pecd"
    os.makedirs(savedir, exist_ok=True)

    url = "https://data.dtu.dk/ndownloader/files/35039785"
    # url = "/Users/tdfalc/Projects/Research/bayesian-regression-markets/analytics/docs/sim08-solar-pecd/solarPV.csv"

    start = dt(2018, 1, 1, 0, 0, tzinfo=pytz.utc)
    # start = dt(2019, 1, 1, 0, 0, tzinfo=pytz.utc)
    end = dt(2019, 12, 31, 23, 30, tzinfo=pytz.utc)
    target_codes = ["UK", "BE", "AT", "GR", "CY", "TR"]
    max_central_agent_lags = 1
    max_support_agent_lags = 1
    polynomial_degree = 1
    num_sellers = len(target_codes) - 1
    policy = NllShapleyPolicy
    forgetting = 0.998
    results = defaultdict(dict)

    colors = classic_colors()
    color_map = {i: colors[i] for i in range(6)}

    target_signals = []

    for target_code in target_codes:
        logger.info(f"Running simulation for: {target_code}")
        cache_location = savedir / f"cache" / target_code
        os.makedirs(cache_location, exist_ok=True)

        (
            support_agent_features,
            central_agent_features,
            target_signal,
        ) = cache(save_dir=cache_location)(
            lambda: pd.read_csv(url).pipe(
                process_raw_data,
                start=start,
                end=end,
                support_agent_codes=list(
                    filter(lambda code: code != target_code, target_codes)
                ),
                target_code=target_code,
                max_central_agent_lags=max_central_agent_lags,
                max_support_agent_lags=max_support_agent_lags,
            )
        )()

        target_signals.append(target_signal)

        batch_data = BatchData(
            dummy_feature=np.ones((len(central_agent_features), 1)),
            central_agent_features=central_agent_features.to_numpy(),
            support_agent_features=support_agent_features.to_numpy(),
            target_signal=target_signal.to_numpy().reshape(-1, 1),
            polynomial_degree=polynomial_degree,
        )

        market_data = MarketData(
            dummy_feature=np.ones((len(central_agent_features), 1)),
            central_agent_features=central_agent_features.to_numpy().reshape(
                -1, 1
            ),
            support_agent_features=support_agent_features.to_numpy(),
            target_signal=target_signal.to_numpy().reshape(-1, 1),
            polynomial_degree=polynomial_degree,
        )

        market = OnlineMarket(
            market_data,
            regression_task=OnlineBayesianLinearRegression(
                regularization=1e-5,
                forgetting=forgetting,
                noise_variance=noise_variance_mle(batch_data),
            ),
            observational=True,
            train_payment=0.5,
            test_payment=1.5,
            burn_in=100,
            likelihood_flattening=forgetting,
        )

        results[target_code][str(policy)] = cache(save_dir=cache_location)(
            lambda policy: market.run(policy, verbose=True)
        )(policy)

    plot_irradiance(target_signals, target_codes, color_map, savedir)
    plot_results(
        results, target_signal.index[:-1], target_codes, color_map, savedir
    )


if __name__ == "__main__":
    main()
