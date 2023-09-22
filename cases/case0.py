"""Batch of simulation cases presented in original regression markets paper.

    Pinson, P., Han, L. & Kazempour, J. Regression markets and application to energy forecasting.
        TOP (2022) (DOI: 10.1007/s11750-022-00631-7)
"""

from mimetypes import init
import numpy as np

from allocation.model import LinearRegression, QuantileRegression
from allocation.state import QuadraticStateTracker, QuantileStateTracker
from allocation.policy import Shapley


def simulations():
    return {
        "1": {
            "data": {
                "model": "linear_transformation",
                "weights": np.array([0.1, -0.3, 0.5, -0.9, 0.2]),
                "polynomial_order": 1,
                "variables": np.random.normal(size=(10000, 4)),
                "additive_noise": np.random.normal(0, 0.3, size=10000),
                "multiplicative_noise": np.random.normal(1, 0, size=(10000, 5)),
            },
            "experiment": {
                "policy": Shapley,
                "batch": True,
                "init": {
                    "forecast_model": LinearRegression(),
                    "dummy_agents": (0,),
                    "central_agents": (1,),
                    "support_agents": (2, 3, 4),
                },
                "run": {
                    "common": {
                        "test_frac": 0,
                    },
                    "train": {
                        "willingness_to_pay": 0.1,
                    },
                },
            },
        },
        "2": {
            "data": {
                "model": "linear_transformation",
                "weights": np.array([0.2, -0.4, 0.6, 0.3, 0.0, 0.0, -0.4, 0.1, 0.0, 0.0]),
                "polynomial_order": 2,
                "variables": np.random.normal(size=(10000, 3)),
                "additive_noise": np.random.normal(0, 0.3, size=10000),
                "multiplicative_noise": np.random.normal(1, 0, size=(10000, 10)),
            },
            "experiment": {
                "policy": Shapley,
                "batch": True,
                "init": {
                    "forecast_model": LinearRegression(),
                    "dummy_agents": (0,),
                    "central_agents": (1,),
                    "support_agents": (2, 3),
                },
                "run": {
                    "common": {
                        "test_frac": 0,
                    },
                    "train": {
                        "willingness_to_pay": 0.1,
                    },
                },
            },
        },
        "3": {
            "data": {
                "model": "linear_transformation_with_lagged_target",
                "weights": np.array([0.1, 0.92, -0.5, 0.3, -0.1]),
                "polynomial_order": 1,
                "variables": np.random.normal(size=(1000, 3)),
                "additive_noise": np.random.normal(0, 0.3, size=1000),
                "multiplicative_noise": np.random.normal(1, 0, size=(1000, 4)),
            },
            "experiment": {
                "policy": Shapley,
                "batch": True,
                "init": {
                    "forecast_model": QuantileRegression(tau=0.5, alpha=0.01),
                    "dummy_agents": (0,),
                    "central_agents": (1,),
                    "support_agents": (2, 3, 4),
                },
                "run": {
                    "common": {
                        "test_frac": 0,
                    },
                    "train": {
                        "willingness_to_pay": 0.1,
                    },
                },
            },
        },
        "4": {
            "data": {
                "model": "linear_transformation_with_lagged_target",
                "weights": np.hstack(
                    [
                        np.linspace(0.8, 0.4, 10000).reshape(-1, 1),
                        np.linspace(0.98, 0.92, 10000).reshape(-1, 1),
                        np.linspace(-0.2, -0.8, 10000).reshape(-1, 1),
                        (lambda a, b: (0.5 + 0.9 * (a * (b - a)) * b**-2))(
                            np.arange(10000) + 1, 0.8 * 10000
                        ).reshape(-1, 1),
                        np.linspace(-0.5, 0, 10000).reshape(-1, 1),
                    ]
                ),
                "polynomial_order": 1,
                "variables": np.random.normal(size=(10000, 3)),
                "additive_noise": np.random.normal(0, 0.3, size=10000),
                "multiplicative_noise": np.random.normal(1, 0, size=(10000, 4)),
            },
            "experiment": {
                "policy": Shapley,
                "batch": False,
                "init": {
                    "forecast_model": LinearRegression(),
                    "dummy_agents": (0,),
                    "central_agents": (1,),
                    "support_agents": (2, 3, 4),
                },
                "run": {
                    "train": {
                        "state": QuadraticStateTracker(
                            forgetting_factor=0.998,
                            initial_weights=np.array([0.8, 0.98, -0.2, 0.5, -0.5]),
                            contributions_burn_in=50,
                            weights_burn_in=50,
                        ),
                        "willingness_to_pay": 0.1,
                    },
                },
            },
        },
        "5": {
            "data": {
                "model": "linear_transformation",
                "weights": np.hstack(
                    [
                        np.linspace(0.8, 0.4, 10000).reshape(-1, 1),
                        np.linspace(0.4, 0.8, 10000).reshape(-1, 1),
                        np.linspace(-0.2, -0.8, 10000).reshape(-1, 1),
                        (lambda a, b: (0.3 + 1.4 * (a * (b - a)) * b**-2))(
                            np.arange(10000) + 1, 0.85 * 10000
                        ).reshape(-1, 1),
                        np.linspace(1.3, 0.7, 10000).reshape(-1, 1),
                    ]
                ),
                "polynomial_order": 1,
                "variables": np.hstack(
                    [
                        np.random.normal(size=(10000, 3)),
                        np.random.uniform(0.5, 1.5, size=(10000, 1)),
                    ]
                ),
                "additive_noise": np.random.normal(0, 0.3, size=10000),
                "multiplicative_noise": np.random.normal(
                    [1, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0.3],
                    size=(10000, 5),
                ),
            },
            "experiment": {
                "policy": Shapley,
                "batch": False,
                "init": {
                    "forecast_model": QuantileRegression(tau=0.5, alpha=0.2),
                    "dummy_agents": (0,),
                    "central_agents": (1,),
                    "support_agents": (2, 3, 4),
                },
                "run": {
                    "train": {
                        "state": QuantileStateTracker(
                            forgetting_factor=0.999,
                            initial_weights=np.array([0.8, 0.4, -0.2, 0.3, 1.3]),
                            contributions_burn_in=101,
                            weights_burn_in=200,
                            alpha=0.2,
                            tau=0.5,
                        ),
                        "willingness_to_pay": 1,
                    }
                },
            },
        },
        "6": {
            "data": {
                "fname": "./data/docs/sc_wind_power.csv",
                "central_agent": 0,
                "num_samples": 20000,
                "polynomial_order": 1,
            },
            "experiment": {
                "policy": Shapley,
                "batch": True,
                "init": {
                    "forecast_model": LinearRegression(),
                    "dummy_agents": (),
                    "central_agents": (1, 2),
                    "support_agents": (3, 4, 5, 6, 7, 8, 9, 10),
                },
                "run": {
                    "common": {
                        "test_frac": 0.5,
                    },
                    "train": {
                        "willingness_to_pay": 50,
                    },
                    "test": {
                        "willingness_to_pay": 150,
                    },
                },
            },
        },
        "7": {
            "data": {
                "fname": "./data/docs/sc_wind_power.csv",
                "central_agent": 0,
                "num_samples": 61300,
                "polynomial_order": 1,
            },
            "experiment": {
                "policy": Shapley,
                "batch": False,
                "init": {
                    "forecast_model": QuantileRegression(tau=0.55, alpha=0.1),
                    "dummy_agents": (),
                    "central_agents": (1, 2),
                    "support_agents": (3, 4, 5, 6, 7, 9),
                },
                "run": {
                    "common": {
                        "state": QuantileStateTracker(
                            forgetting_factor=0.9995,
                            initial_weights=np.zeros(8),
                            contributions_burn_in=500,
                            weights_burn_in=501,
                            alpha=0.1,
                            tau=0.55,
                        ),
                    },
                    "train": {
                        "willingness_to_pay": 20,
                    },
                    "test": {
                        "willingness_to_pay": 80,
                    },
                },
            },
        },
    }
