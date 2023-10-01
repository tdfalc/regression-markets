from pathlib import Path
from typing import Sequence, Any
import os
from itertools import cycle

from joblib import Parallel, delayed
import numpy as np
from matplotlib import cm
import matplotlib as mpl
from matplotlib import pyplot as plt
from tqdm import tqdm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition

from market.task import RobustBayesianLinearRegression
from common.log import create_logger
from common.utils import cache, tqdm_joblib
from analytics.helpers import (
    save_figure,
    set_plot_style,
    # dark2_color_palette,
    # create_custom_colormap,
)


def sample_input(num_features: int) -> np.ndarray:
    mean, covariance = np.zeros(num_features), np.eye(num_features)
    x = np.random.multivariate_normal(mean, covariance)
    return np.append(np.ones(1), x)


def generate_sample(noise_variance: float, coefficients: np.ndarray):
    for c in coefficients:
        x = sample_input(len(c) - 1)
        y = np.dot(c, x) + np.random.normal(0, np.sqrt(noise_variance))
        yield x, y


def plot_coefficients(
    experiments: dict,
    results: dict,
    forgetting_factors: Sequence,
    agent: int,
    savedir: Path,
    burn_in: int = 50,
):
    # colors = dark2_color_palette(4)
    # colors = [
    #     "#C44F51",
    #     "#55A868",
    #     "#4C72B0",
    #     "#CCB974",
    # ]
    # cmap_custom = create_custom_colormap(colors[2], "#FFFFFF")
    # colors = cycle(colors)
    vmin, vmax = np.min(forgetting_factors), np.max(forgetting_factors)
    norm = lambda vmax: mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    # smap = cm.ScalarMappable(norm=norm(vmax + 0.03), cmap=cmap_custom)
    # smap_fake = cm.ScalarMappable(norm=norm(vmax), cmap=cmap_custom)

    fig, axs = plt.subplots(1, len(experiments), sharey=True, figsize=(8, 3))

    for i, ((experiment_title, experiment_config), ax) in enumerate(
        zip(experiments.items(), axs.flatten())
    ):
        estimated_coefficients = results[experiment_title]
        coefficients = experiment_config["coefficients"]
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

        # for j in range(coefficients.shape[1]):
        ax.plot(
            coefficients[burn_in:, agent],
            # color=next(colors),
            # color="k",
            # zorder=1 if j != agent else 2,
            # label=f"$w_{agent#}$",
            lw=1.6,
            # alpha=0.3 if j != agent else 0.3,
        )
        # ax.grid(alpha=0.3)

        # if j == agent:
        for ff in forgetting_factors:
            data = estimated_coefficients[ff][:, burn_in:, agent]
            ax.plot(
                data.mean(axis=0),
                zorder=2,
                # color=smap.to_rgba(ff),
                lw=1.6,
                ls=(0, (5, 4)),  # (length, spacing)
            )

        ax.set_xlabel("Time Step")
        # if i == 0:
        # ax.legend()

        ax.set_ylabel("Coefficient Value")
        ax.yaxis.set_tick_params(labelbottom=True)

    ax = axs.flatten()[-2]
    axins = inset_axes(ax, width="5%", height="100%", loc="right", borderpad=-4)
    ip = InsetPosition(ax, [0.5, 1.1, 1.2, 0.09])  # posx, posy, width, height
    axins.set_axes_locator(ip)
    # cb = fig.colorbar(
    #     smap_fake,
    #     cax=axins,
    #     label=r"$\tau$",
    #     orientation="horizontal",
    #     # labelpad=-40,
    # )
    # cb.set_label(r"$\tau$", labelpad=7)
    axins.xaxis.set_ticks_position("top")
    axins.xaxis.set_label_position("top")

    save_figure(fig, savedir, f"estimated_coefficients")


def main():
    logger = create_logger(__name__)
    logger.info("Running nonstationary processes analysis")

    set_plot_style()

    savedir = Path(__file__).parent / "docs/sim03-nonstationary-processes"
    os.makedirs(savedir, exist_ok=True)

    noise_variance = 1
    sample_size = 10000
    num_samples = 100
    regularization = 1e-5
    forgetting_factors = [
        0.92,
        0.93,
        0.94,
        0.95,
        0.96,
        0.97,
        0.98,
        0.99,
        0.995,
        0.999,
        0.9999,
        0.99999,
        1,
    ]

    experiments = {
        # "stationary": {
        #     "coefficients": np.tile(
        #         np.array([0, -0.2, 0.6, 0.3]), (sample_size, 1)
        #     )
        # },
        "smooth_nonstationarity": {
            "coefficients": np.hstack(
                [
                    np.linspace(0, 0, sample_size).reshape(-1, 1),
                    np.linspace(-0.2, -0.2, sample_size).reshape(-1, 1),
                    np.linspace(0.6**0.5, 0, sample_size).reshape(-1, 1) ** 2,
                    np.linspace(0.3, 0.3, sample_size).reshape(-1, 1),
                ]
            ),
        },
        "step_nonstationarity": {
            "coefficients": np.concatenate(
                (
                    np.tile(
                        np.array([0, -0.2, 0.6, 0.3]),
                        (int(sample_size / 2.2), 1),
                    ),
                    np.tile(
                        np.array([0, -0.2, 0, 0.3]),
                        (int(sample_size / 1.8) - int(sample_size / 2.2), 1),
                    ),
                    np.tile(
                        np.array([0, -0.2, 0, 0.3]),
                        (
                            sample_size - int(sample_size / 1.8),
                            1,
                        ),
                    ),
                ),
                axis=0,
            ),
        },
    }

    results = {}
    for experiment_title, experiment_config in experiments.items():
        logger.info(f"Running experiment: {experiment_title}")

        coefficients = experiment_config["coefficients"]

        cache_location = savedir / "cache" / experiment_title
        os.makedirs(cache_location, exist_ok=True)

        @cache(cache_location)
        def _run_experiment():
            def _simulate():
                estimated_coefficients = {
                    ff: np.zeros((sample_size, coefficients.shape[1]))
                    for ff in forgetting_factors
                }

                models = {
                    ff: RobustBayesianLinearRegression(
                        regularization=regularization,
                        noise_variance=noise_variance,
                        forgetting=ff,
                    )
                    for ff in forgetting_factors
                }

                for i, (xi, yi) in enumerate(
                    generate_sample(noise_variance, coefficients)
                ):
                    for ff in models.keys():
                        indices = np.arange(coefficients.shape[1])
                        models[ff].update_posterior(xi, yi, indices=indices)
                        posterior = models[ff].get_posterior(indices)
                        estimated_coefficients[ff][i, :] = posterior.mean

                return estimated_coefficients

            with tqdm_joblib(
                tqdm(desc="Simulation progress", total=num_samples)
            ) as _:
                estimated_coefficients = Parallel(n_jobs=-1)(
                    delayed(_simulate)() for run in range(num_samples)
                )

            return {
                ff: np.stack(e[ff] for e in estimated_coefficients)
                for ff in forgetting_factors
            }

        results[experiment_title] = _run_experiment()

    plot_coefficients(
        experiments,
        results,
        [
            # ###0.92,
            # 0.93,
            # 0.94,
            # 0.95,
            # 0.96,
            # 0.97,
            # 0.98,
            # 0.99,
            # 0.995,
            # 0.999,
            # 0.9999,
            # 0.99999,
            1,
        ],
        agent=2,
        savedir=savedir,
    )


if __name__ == "__main__":
    np.random.seed(123)
    main()
