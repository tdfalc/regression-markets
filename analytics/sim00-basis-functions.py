from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from tfds.plotting import use_tex, prettify

from regression_markets.common.log import create_logger
from analytics.helpers import save_figure, set_style


use_tex()
set_style()


class BasisFunctions:
    def __init__(self, X: np.ndarray) -> None:
        self.X = X

    def trigonometric(
        self, offset: float, freq: float, scale: float
    ) -> np.ndarray:
        return offset + scale * np.cos(freq * np.pi * self.X)

    def sigmoidal(self, mean: float, sdev: float) -> np.ndarray:
        return 1 / (1 + np.exp(-(self.X - mean) / sdev))

    def rq(self, mean: float, variance: float, alpha: float) -> np.ndarray:
        return (1 + 0.5 * (self.X - mean) ** 2 / (alpha * variance)) ** -alpha

    def polynomial(self, power: int) -> np.ndarray:
        return self.X**power


def main() -> None:
    logger = create_logger(__name__)
    logger.info("Running basis functions analysis")

    savedir = Path(__file__).parent / "docs" / "sim00-basis-functions"
    savedir.mkdir(parents=True, exist_ok=True)

    X = np.linspace(-1, 1, 100).reshape(-1, 1)
    basis_functions = BasisFunctions(X)

    experiments = {
        "sigmoidal": {
            "basis_function": "sigmoidal",
            "kwargs": [
                {"mean": -2 / 3, "sdev": 0.05},
                {"mean": -2 / 9, "sdev": 0.05},
                {"mean": 2 / 9, "sdev": 0.05},
                {"mean": 2 / 3, "sdev": 0.05},
            ],
        },
        "polynomial": {
            "basis_function": "polynomial",
            "kwargs": [
                {"power": 0},
                {"power": 1},
                {"power": 2},
                {"power": 3},
            ],
        },
        "rational_quadratic": {
            "basis_function": "rq",
            "kwargs": [
                {"mean": 0, "variance": 0.03, "alpha": 3},
                {"mean": 0, "variance": 0.2, "alpha": 3},
                {"mean": 0, "variance": 0.7, "alpha": 3},
                {"mean": 0, "variance": 2, "alpha": 3},
            ],
        },
        # "trigonometric": {
        #     "basis_function": "trigonometric",
        #     "kwargs": [
        #         {"offset": 0, "scale": 1, "freq": 0.5},
        #         {"offset": 2.2, "scale": 0.5, "freq": 3},
        #         {"offset": 4, "scale": 0.2, "freq": 10},
        #         {"offset": 6, "scale": 1, "freq": 5},
        #     ],
        # },
        "trigonometric": {
            "basis_function": "trigonometric",
            "kwargs": [
                {"offset": 0, "scale": 0.15, "freq": 0.5},
                {"offset": 0.35, "scale": 0.15, "freq": 3},
                {"offset": 0.65, "scale": 0.05, "freq": 15},
                {"offset": 0.9, "scale": 0.1, "freq": 5},
            ],
        },
    }

    # Create a 2x2 grid for subplots
    fig, axs = plt.subplots(2, 2, figsize=(8, 6))

    # Use a colormap with 4 colors (each experiment has 4 curves)
    cmap = plt.get_cmap("viridis", 4).colors

    for (experiment_title, experiment_config), ax in zip(
        experiments.items(), axs.flatten()
    ):
        for i, kwargs in enumerate(experiment_config["kwargs"]):
            basis_func = getattr(
                basis_functions, experiment_config["basis_function"]
            )
            phi = basis_func(**kwargs)
            ax.plot(X, phi, color=cmap[i], lw=1)
        ax.set_xlabel(r"$x_t$")
        ax.set_ylabel(r"$\varphi(x_t)$")
        title = " ".join(
            word.capitalize() for word in experiment_title.split("_")
        )
        ax.set_title(title)
        ax.set_xlim(-1, 1)
        ax.set_ylim(bottom=0, top=1.05)
        prettify(ax=ax, legend=False, legend_loc="lower right")

    save_figure(fig, savedir, "basis_functions")


if __name__ == "__main__":
    np.random.seed(42)
    main()
