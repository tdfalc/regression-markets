from pathlib import Path
from typing import Dict
import os

import cloudpickle
import numpy as np

from data.synthetic import DataGenerator
from data.south_carolina import build_data
from common.plot import (
    plot_payments_evolution,
    plot_target,
    plot_weights_evolution,
    plot_payments_final,
)
from cases import *
from common.utils import file_cache, get_logger

logger = get_logger(__name__)


def save_results(results: np.ndarray, save_location: Path):
    with open(save_location, "wb") as f:
        cloudpickle.dump(results, f)


def _run_simulation(simulation: Dict, id: int, case: str, directory: Path):

    data = simulation["data"]
    experiment = simulation["experiment"]
    experiment["init"]["polynomial_order"] = data["polynomial_order"]
    batch = experiment["batch"]

    if (fname := data.get("fname")) is not None:
        X, y = build_data(Path(fname), data["central_agent"], data["num_samples"])

    else:
        generator = DataGenerator(
            variables=data["variables"],
            additive_noise=data["additive_noise"],
            multiplicative_noise=data["multiplicative_noise"],
        )
        model = getattr(generator, data["model"])
        X, y = model(data["weights"], data["polynomial_order"])

        if not batch:
            plot_weights_evolution(
                data["weights"],
                f"{case.capitalize()} (sim {id}): Population Parameters",
                save_location=directory / "weights_evolution_population.png",
            )

    plot_target(
        y,
        f"{case.capitalize()} (sim {id}): Target Variable",
        save_location=directory / "target.png",
    )

    run = experiment["run"]
    policy = experiment["policy"](X, y, **experiment["init"])
    common_kwargs = run.get("common", {})
    market = policy.run_batch_market if batch else policy.run_online_market

    results = {}
    for method in ("train", "test"):
        try:
            run[method].update(train=False if method == "test" else True)
            if batch and (method == "test"):
                run[method].update(all_weights=results["train"]["weights"])

            results[method] = market(**(common_kwargs | run[method]))

            if not batch:
                plot_weights_evolution(
                    results[method]["weights"],
                    f"{case.capitalize()} (sim {id}): Estimated Parameters",
                    save_location=directory / f"weights_evolution_estimated_{method}.png",
                )
                plot_payments_evolution(
                    results[method]["payments"].cumsum(axis=0),
                    f"{case.capitalize()} (sim {id}): Cumulative Payments",
                    save_location=directory / f"payments_evolution_cumulative_{method}.png",
                )
                plot_payments_evolution(
                    results[method]["payments"],
                    f"{case.capitalize()} (sim {id}): Payments",
                    save_location=directory / f"payments_evolution_{method}.png",
                )

            plot_payments_final(
                results[method]["payments"]
                if batch
                else results[method]["payments"].cumsum(axis=0)[-1],
                f"{case.capitalize()} (sim {id}): Payments Final",
                save_location=directory / f"payments_final_{method}.png",
            )

        except KeyError:
            pass

    return results


def run_case(case: str, directory: Path, use_cache: bool = False):
    try:
        simulations = eval(case)()
    except NameError:
        raise Exception(f"Specified case does not exist in ./cases directory: {case}")

    for id, simulation in simulations.items():
        logger.info(f"Running {case} (sim {id})")
        simulation_directory = directory / f"{case}/simulation_{id}"
        os.makedirs(simulation_directory, exist_ok=True)
        _ = file_cache(save_location=simulation_directory / "results.pkl", use_cache=use_cache)(
            lambda simulation, id, case, directory: _run_simulation(simulation, id, case, directory)
        )(simulation, id, case, simulation_directory)
