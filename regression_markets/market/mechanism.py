from typing import Type, Dict

import numpy as np
from tqdm import tqdm

from regression_markets.market import data
from regression_markets.market.policy import SemivaluePolicy
from regression_markets.market.task import Task
from regression_markets.common.utils import chain_combinations


class Market:
    def __init__(
        self,
        market_data: data.MarketData,
        regression_task: Task,
        observational: bool = True,
        train_payment: float = 1,
        test_payment: float = 1,
    ) -> None:
        self.market_data = market_data
        self.regression_task = regression_task
        self.observational = observational
        self.train_payment = train_payment
        self.test_payment = test_payment

    def _run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        payment: float,
        policy: Type[SemivaluePolicy],
    ) -> Dict[str, np.ndarray]:
        return policy(
            active_agents=self.market_data.active_agents,
            baseline_agents=self.market_data.baseline_agents,
            polynomial_degree=self.market_data.degree,
            regression_task=self.regression_task,
            observational=self.observational,
        ).run(X, y, payment)

    def _precalculate_posteriors(self, X: np.ndarray, y: np.ndarray) -> None:
        num_features = X.shape[1]
        if self.observational:
            for indices in chain_combinations(
                np.arange(num_features), 1, num_features
            ):
                self.regression_task.update_posterior(X, y, indices)
        else:
            self.regression_task.update_posterior(X, y, np.arange(num_features))


class BatchMarket(Market):
    def __init__(
        self,
        market_data: data.BatchData,
        regression_task: Task,
        observational: bool = True,
        train_payment: float = 1,
        test_payment: float = 1,
    ) -> None:
        super().__init__(
            market_data=market_data,
            regression_task=regression_task,
            observational=observational,
            train_payment=train_payment,
            test_payment=test_payment,
        )
        self._precalculate_posteriors(
            self.market_data.X_train, self.market_data.y_train
        )

    def run(self, policy: Type[SemivaluePolicy]) -> Dict[str, np.ndarray]:
        results = {}
        for stage, payment, X, y in zip(
            ("train", "test"),
            (self.train_payment, self.test_payment),
            (self.market_data.X_train, self.market_data.X_test),
            (self.market_data.y_train, self.market_data.y_test),
        ):
            results[stage] = self._run(X, y, payment, policy)

        return results


class OnlineMarket(Market):
    def __init__(
        self,
        market_data: data.MarketData,
        regression_task: Task,
        burn_in: int,
        likelihood_flattening: float,
        observational: bool = True,
        train_payment: float = 1,
        test_payment: float = 1,
    ) -> None:
        super().__init__(
            market_data=market_data,
            regression_task=regression_task,
            observational=observational,
            train_payment=train_payment,
            test_payment=test_payment,
        )
        self.burn_in = burn_in
        self.likelihood_flattening = likelihood_flattening
        self.num_active_agents = len(self.market_data.active_agents)
        self.num_runs = len(self.market_data.X) - 1
        self.X = self.market_data.X
        self.y = self.market_data.y

    def _init_empty_results(self) -> None:
        return {
            "allocations": np.zeros((self.num_runs, self.num_active_agents)),
            "contributions": np.zeros((self.num_runs, self.num_active_agents)),
            "payments": np.zeros((self.num_runs, self.num_active_agents)),
        }

    def _run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        payment: float,
        policy: Type[SemivaluePolicy],
    ) -> Dict[str, np.ndarray]:
        return policy(
            active_agents=self.market_data.active_agents,
            baseline_agents=self.market_data.baseline_agents,
            polynomial_degree=self.market_data.degree,
            regression_task=self.regression_task,
            observational=self.observational,
        ).run(X, y, payment)

    def run(
        self, policy: Type[SemivaluePolicy], verbose: bool = False
    ) -> Dict[str, np.ndarray]:
        results = {
            "train": self._init_empty_results(),
            "test": self._init_empty_results(),
        }

        indices = range(self.num_runs)
        iterator = (
            tqdm(indices, desc="Running market", unit="item")
            if verbose
            else indices
        )

        for i in iterator:
            X_train, X_test = self.X[i : i + 1], self.X[i + 1 : i + 2]
            y_train, y_test = self.y[i : i + 1], self.y[i + 1 : i + 2]
            self._precalculate_posteriors(X_train, y_train)
            if i >= self.burn_in:
                for key, X, y, payment in (
                    ("train", X_train, y_train, self.train_payment),
                    ("test", X_test, y_test, self.test_payment),
                ):
                    output = self._run(X, y, payment, policy)
                    indices = np.array([0, 1])

                    for attribute in (
                        "contributions",
                        "allocations",
                        "payments",
                    ):
                        results[key][attribute][i, :] = (
                            output[attribute] * (1 - self.likelihood_flattening)
                            + self.likelihood_flattening
                            * results[key][attribute][i - 1, :]
                        )

        return results
