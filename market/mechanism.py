import numpy as np

from market import data
from market.policy import ShapleyAttributionPolicy
from market.task import Task
from common.utils import chain_combinations


class BatchMarket:
    def __init__(
        self,
        market_data: data.MarketData,
        regression_task: Task,
        train_payment: float = 1,
        test_payment: float = 1,
    ):
        self.market_data = market_data
        self.regression_task = regression_task
        self.train_payment = train_payment
        self.test_payment = test_payment
        self._precalculate_posteriors(
            self.market_data.X_train, self.market_data.y_train
        )

    def _precalculate_posteriors(self, X: np.ndarray, y: np.ndarray):
        num_features = X.shape[1]
        for indices in chain_combinations(np.arange(num_features), 1, num_features):
            self.regression_task.update_posterior(X, y, indices)

    def run(self):
        results = {}
        for stage, payment, X, y in zip(
            ("train", "test"),
            (self.train_payment, self.test_payment),
            (self.market_data.X_train, self.market_data.X_test),
            (self.market_data.y_train, self.market_data.y_test),
        ):
            results[stage] = ShapleyAttributionPolicy(
                active_agents=self.market_data.active_agents,
                baseline_agents=self.market_data.baseline_agents,
                regression_task=self.regression_task,
            ).run(X, y, payment)

        return results
