from typing import Type, Dict

import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

from market import data
from market.policy import SemivaluePolicy
from market.task import Task
from common.utils import chain_combinations


class Market:
    def __init__(
        self,
        market_data: data.MarketData,
        regression_task: Task,
        observational: bool = True,
        train_payment: float = 1,
        test_payment: float = 1,
    ):
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
    ):
        return policy(
            active_agents=self.market_data.active_agents,
            baseline_agents=self.market_data.baseline_agents,
            polynomial_degree=self.market_data.degree,
            regression_task=self.regression_task,
            observational=self.observational,
        ).run(X, y, payment)

    def _precalculate_posteriors(self, X: np.ndarray, y: np.ndarray):
        num_features = X.shape[1]
        for indices in chain_combinations(np.arange(num_features), 1, num_features):
            self.regression_task.update_posterior(X, y, indices)


class BatchMarket(Market):
    def __init__(
        self,
        market_data: data.BatchData,
        regression_task: Task,
        observational: bool = True,
        train_payment: float = 1,
        test_payment: float = 1,
    ):
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

    def run(self, policy: Type[SemivaluePolicy]):
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
    ):
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

    def _init_empty_results(self):
        return {
            "allocations": np.zeros((self.num_runs, self.num_active_agents)),
            "contributions": np.zeros((self.num_runs, self.num_active_agents)),
            "payments": np.zeros((self.num_runs, self.num_active_agents)),
            "loss_gc": np.zeros((self.num_runs, 1)),
            "loss_buyer": np.zeros((self.num_runs, 1)),
        }

    # def _run(self, X: np.ndarray, y: np.ndarray, policy: Type[SemivaluePolicy]):
    #     return policy(
    #         self.market_data.active_agents,
    #         self.market_data.baseline_agents,
    #         self.market_data.degree,
    #         self.regression_task,
    #     ).run(X, y, 1)

    def _run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        payment: float,
        policy: Type[SemivaluePolicy],
    ):
        return policy(
            active_agents=self.market_data.active_agents,
            baseline_agents=self.market_data.baseline_agents,
            polynomial_degree=self.market_data.degree,
            regression_task=self.regression_task,
            observational=self.observational,
        ).run(X, y, payment)

    def run(self, policy: Type[SemivaluePolicy], verbose: bool = False):
        from scipy import stats

        results = {
            "train": self._init_empty_results(),
            "test": self._init_empty_results(),
        }
        # for i, (X_train, X_test, y_train, y_test) in enumerate(zip(self.X[:-1], self.X[1:], self.y[:-1], self.y[1:])):
        indices = range(self.num_runs)
        iterator = (
            tqdm(indices, desc="Running market", unit="item") if verbose else indices
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

                    if self.observational:
                        output["loss_buyer"] = self.regression_task.calculate_loss(
                            X[:, [0, 1]],
                            y,
                            self.regression_task.get_posterior(indices),
                            self.regression_task.get_noise_variance(indices),
                        )
                        indices = np.arange(X.shape[1])
                        output["loss_gc"] = self.regression_task.calculate_loss(
                            X,
                            y,
                            self.regression_task.get_posterior(indices),
                            self.regression_task.get_noise_variance(indices),
                        )

                    else:
                        posterior = self.regression_task.get_posterior(
                            np.arange(X.shape[1])
                        )
                        mean = posterior.mean[np.sort(indices)]
                        cov = posterior.cov[:, np.sort(indices)][np.sort(indices), :]

                        posterior_buyer = stats.multivariate_normal(mean, cov)

                        output["loss_buyer"] = self.regression_task.calculate_loss(
                            X[:, [0, 1]],
                            y,
                            posterior_buyer,
                            self.regression_task.get_noise_variance(indices),
                        )
                        indices = np.arange(X.shape[1])
                        output["loss_gc"] = self.regression_task.calculate_loss(
                            X,
                            y,
                            posterior,
                            self.regression_task.get_noise_variance(indices),
                        )

                    for attribute in (
                        "contributions",
                        "allocations",
                        "payments",
                        "loss_gc",
                        "loss_buyer",
                    ):
                        results[key][attribute][i, :] = (
                            output[attribute] * (1 - self.likelihood_flattening)
                            + self.likelihood_flattening
                            * results[key][attribute][i - 1, :]
                        )

        return results
