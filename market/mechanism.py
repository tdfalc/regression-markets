from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt


from market import data
from market.task import Task
from common.utils import chain_combinations
from market.impute import imputer_factory
from market.policy import ShapleyAttributionPolicy
from common.utils import chain_combinations


class Market:
    def __init__(self, market_data: data.BatchData, regression_task: Task):
        self.market_data = market_data
        self.regression_task = regression_task

        self.num_features = self.market_data.X_train.shape[1]

    def run(
        self, imputation_methods: Sequence, missing_probabilities: float, payment: float
    ):
        X_train, X_test = self.market_data.X_train, self.market_data.X_test
        y_train, y_test = self.market_data.y_train, self.market_data.y_test

        regression_task = self.regression_task()
        indices = np.arange(X_train.shape[1])

        for indices in chain_combinations(
            np.arange(self.num_features), 1, self.num_features
        ):
            regression_task.fit(X_train, y_train, indices)

        missing_indicator = (
            np.random.rand(len(X_test), len(missing_probabilities))
            < missing_probabilities
        )

        imputers = [
            imputer_factory(self.market_data, method) for method in (imputation_methods)
        ]

        def zeros(size: int):
            return np.zeros((len(X_test), size))

        losses = {method: zeros(1) for method in imputation_methods}
        primary_market_payments = {
            method: zeros(self.market_data.num_support_agent_features)
            for method in imputation_methods
        }
        secondary_market_payments = {
            method: zeros(self.market_data.num_support_agent_features)
            for method in imputation_methods
        }

        for i in range(len(X_test)):
            x_test = X_test[i : i + 1]
            for method, imputer in zip(imputation_methods, imputers):
                x_imputed_mean, x_imputed_covariance = imputer.impute(
                    x_test, missing_indicator[i]
                )

                losses[method][i, :] = regression_task.calculate_loss(
                    x_imputed_mean,
                    y_test[i],
                    indices,
                    X_covariance=x_imputed_covariance,
                )

                attribution_policy = ShapleyAttributionPolicy(
                    active_agents=self.market_data.active_agents,
                    baseline_agents=self.market_data.baseline_agents,
                    regression_task=regression_task,
                )

                (_, _, primary_payments) = attribution_policy.run(
                    x_imputed_mean,
                    y_test[i],
                    X_covariance=x_imputed_covariance,
                    payment=payment,
                )

                if imputer.has_secondary_market:
                    secondary_payments = imputer.clear_secondary_market(
                        x_test, missing_indicator[i], primary_payments
                    )
                    secondary_market_payments[method][i, :] = secondary_payments
                ## The revenue should not be 0 in the case of mean imputation, any
                # benefit of the mean imputatin should beallocated to the owner..
                if method.value != "no":
                    for j, is_missing in enumerate(missing_indicator[i]):
                        if is_missing:
                            primary_payments[j] = 0

                primary_market_payments[method][i, :] = primary_payments

        return losses, primary_market_payments, secondary_market_payments
