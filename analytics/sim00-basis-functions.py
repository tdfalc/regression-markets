from pathlib import Path
import os

import numpy as np
from matplotlib import pyplot as plt

from regression_markets.common.log import create_logger
from analytics.helpers import save_figure


class BasisFunctions:
    def __init__(self, X: np.ndarray):
        self.X = X

    def trigonometric(self, func: str):
        return 0.5 + eval(func)(2 * np.pi * self.X)

    def constant(self, constant: float):
        return np.repeat(constant, len(self.X))

    def linear(self, coefficient: float):
        return coefficient * self.X

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

    experiments = {
        "constant": {
            "basis_function": "constant",
            "kwargs": [
                {"constant": 0.2},
                {"constant": 0.3},
                {"constant": 0.8},
            ],
        },
        "linear": {
            "basis_function": "linear",
            "kwargs": [
                {"coefficient": 0.2},
                {"coefficient": -0.3},
                {"coefficient": 0.8},
            ],
        },
        "polynomial": {
            "basis_function": "polynomial",
            "kwargs": [
                {"power": 1},
                {"power": 2},
                {"power": 3},
            ],
        },
        "gaussian": {
            "basis_function": "gaussian",
            "kwargs": [
                {"mean": 0.2, "variance": 0.3},
                {"mean": 0.5, "variance": 0.03},
                {"mean": -0.5, "variance": 1},
            ],
        },
        "trigonometric": {
            "title": "Trigonometric",
            "basis_function": "trigonometric",
            "kwargs": [
                {"func": "np.sin"},
                {"func": "np.cos"},
                # {"func": np.tan},
            ],
        },
        "rational_quadratic": {
            "basis_function": "rq",
            "kwargs": [
                {"mean": 0.5, "variance": 0.3, "alpha": 0.1},
                {"mean": 0.5, "variance": 0.3, "alpha": 0.2},
                {"mean": 0.5, "variance": 0.3, "alpha": 0.7},
            ],
        },
    }

    fig, axs = plt.subplots(3, 2, figsize=(8, 9))

    for (experiment_title, experiment_config), ax in zip(
        experiments.items(), axs.flatten()
    ):
        for kwargs in experiment_config["kwargs"]:
            phi = getattr(basis_functions, experiment_config["basis_function"])(
                **kwargs
            )
            ax.plot(X, phi, label=str(kwargs))
        ax.set_xlabel("x")
        ax.set_ylabel(r"$\phi(x)$")
        ax.set_title(experiment_title)
        ax.legend()

    save_figure(fig, savedir, "basis_functions")


if __name__ == "__main__":
    np.random.seed(42)
    main()
