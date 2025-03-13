from pathlib import Path
import os

import numpy as np
from matplotlib import pyplot as plt

from regression_markets.common.log import create_logger
from analytics.helpers import save_figure
from tfds.plotting import use_tex, prettify

use_tex()


class BasisFunctions:
    def __init__(self, X: np.ndarray):
        self.X = X

    def trigonometric(self, offset, freq, scale: str):
        return offset + scale * np.sin(freq * np.pi * self.X)

    def constant(self, constant: float):
        return np.repeat(constant, len(self.X))

    def sigmoidal(self, mean: float, sdev: float):
        return 1 / (1 + np.exp(-(self.X - mean) / sdev))

    def gaussian(self, mean: float, variance: float):
        return np.exp(-0.5 * (self.X - mean) ** 2 / variance)

    def rq(self, mean: float, variance: float, alpha: float):
        return (1 + 0.5 * (self.X - mean) ** 2 / (alpha * variance)) ** -alpha

    def polynomial(self, power: int):
        return self.X**power


def main() -> None:
    logger = create_logger(__name__)
    logger.info("Running basis functions analysis")

    savedir = Path(__file__).parent / "docs/sim00-basis-functions"
    os.makedirs(savedir, exist_ok=True)

    X = np.linspace(-1, 1, 100).reshape(-1, 1)
    basis_functions = BasisFunctions(X)
    P = 4

    experiments = {
        # "constant": {
        #     "basis_function": "constant",
        #     "kwargs": [
        #         {"constant": 0.2},
        #         {"constant": 0.7},
        #         {"constant": 0.8},
        #     ],
        # },
        "sigmoidal": {
            "basis_function": "sigmoidal",
            "kwargs": [
                {"mean": m, "sdev": 0.05}
                for m in np.linspace(-2 / 3, 2 / 3, P)
                # for m in np.linspace(-1, 1, P)
            ],
        },
        "polynomial": {
            "basis_function": "polynomial",
            "kwargs": [
                {"power": p}
                # for m in np.linspace(-2 / 3, 2 / 3, 11)
                for p in np.arange(0, P)
            ],
        },
        "gaussian": {
            "basis_function": "gaussian",
            # "kwargs": [
            #     {"mean": -0.0, "variance": 0.9},
            #     {"mean": -0.3, "variance": 0.3},
            #     {"mean": -0.5, "variance": 0.1},
            # ],
            # "kwargs": [
            #     {"mean": 0, "variance": 0.9},
            #     {"mean": 0, "variance": 0.3},
            #     {"mean": 0, "variance": 0.1},
            # ],
            "kwargs": [
                {"mean": 0, "variance": v}
                # for m in np.linspace(-2 / 3, 2 / 3, 11)
                for v in [0.05, 0.3, 0.7, 2]
            ],
        },
        "trigonometric": {
            "title": "Trigonometric",
            "basis_function": "trigonometric",
            "kwargs": [
                {"offset": o, "scale": s, "freq": f}
                # for m in np.linspace(-2 / 3, 2 / 3, 11)
                for (o, s, f) in zip(
                    [-4, -1.5, 1.5, 4],
                    [1, 1, 1, 1],
                    [0.5, 1, 2.5, 5],
                )
            ],
        },
        # "rational_quadratic": {
        #     "basis_function": "rq",
        #     "kwargs": [
        #         {"mean": 0, "variance": 0.3, "alpha": 0.1},
        #         {"mean": 0, "variance": 0.3, "alpha": 0.2},
        #         {"mean": 0, "variance": 0.3, "alpha": 0.7},
        #     ],
        # },
    }

    fig, axs = plt.subplots(2, 2, figsize=(8, 6))
    # fig, axs = plt.subplots(2, 3, figsize=(8, 5))

    def make_label(params: dict) -> str:
        """Return a nicer label like 'constant=0.2, another=0.3'."""
        # Sort to have a consistent order of keys in the label
        items = sorted(params.items())
        # Format each "key=value" pair, then join
        string = ", ".join(f"{k}={v}" for k, v in items)
        string = string.replace("mean", "$\mu$")
        string = string.replace("variance", "$\sigma^2$")
        string = string.replace("constant", "$c$")
        string = string.replace("coefficient", "$w$")
        string = string.replace("alpha", r"$\alpha$")
        string = string.replace("power", r"$\alpha$")
        string = string.replace("np.", "")
        string = string.replace("func=", "")
        string = string.replace("sin", "Sine")
        string = string.replace("cos", "Cosine")
        return string

    c = [
        # "#000000",  # Black
        "#F5793A",  # Orange
        "#4BA6FB",  # Modified blue (original was #85C0F9)
        "#A95AA1",  # Pink
        "#689948",  # Green: not optimal for colour-blind people
    ] * 10
    # c = ["lime", "b", "y", "purple", "teal", "r"] * 10
    # c = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#e6ab02"] * 2
    # c = [
    #     "#2F3EEA",
    #     "#1FD082",
    #     "#030F4F",
    #     "#F6D04D",
    #     "#FC7634",
    # ]
    import matplotlib as mpl

    # c = plt.get_cmap("viridis", 4).colors  # 11 discrete colors

    for (experiment_title, experiment_config), ax in zip(
        experiments.items(), axs.flatten()
    ):
        for i, kwargs in enumerate(experiment_config["kwargs"]):
            phi = getattr(basis_functions, experiment_config["basis_function"])(
                **kwargs
            )
            ax.plot(X, phi, label=make_label(kwargs), color=c[i], lw=1)
        ax.set_xlabel(r"$x_t$")
        ax.set_ylabel(r"$\varphi(x_t)$")
        ax.set_title(
            " ".join(word.capitalize() for word in experiment_title.split("_"))
        )
        ax.set_xlim(-1, 1)
        ax.set_ylim(bottom=0)

        prettify(ax=ax, legend=False, legend_loc="lower right")

    save_figure(fig, savedir, "basis_functions")


if __name__ == "__main__":
    np.random.seed(42)
    main()
