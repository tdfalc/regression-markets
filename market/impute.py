from enum import Enum

import numpy as np

from common.utils import chain_combinations
from market.task import (
    MaximumLikelihoodLinearRegression,
    BayesianLinearRegression,
    GaussianProcessLinearRegression,
)
from market.data import BatchData
from market.policy import ShapleyAttributionPolicy


class ImputationMethod(Enum):
    no = "no"
    mean = "mean"
    ols = "ols"
    mle = "mle"
    blr = "blr"
    gpr = "gpr"


class Imputer:
    def __init__(self, market_data: BatchData):
        self.market_data = market_data
        self.X_train = self.market_data.X_train
        self.y_train = self.market_data.y_train
        self.num_central_agent_features = self.market_data.num_central_agent_features
        self._indices = np.arange(self.X_train.shape[1])
        self._support_agent_indices = self._indices[
            1 + self.num_central_agent_features :
        ]
        self.has_secondary_market = False

    def _impute(self, i: int, mean: np.ndarray, not_missing: np.ndarray):
        raise NotImplementedError

    def impute(self, x: np.ndarray, missing_indicator: np.ndarray):
        """Impute missing values.

        Args:
            x (np.ndarray): the index point.
            missing_indicator (np.ndarray): binary array containing a 1 if the feature value
                is missing, else a 0.
        Returns:
            Tuple[np.ndarray, np.ndarray]: posterior predictive mean and standard deviation.
        """
        # Initialise covariance to zero, as only some methods will provide
        # uncertainty estimates for the imputation.
        mean, covariance = x.copy(), np.eye(x.shape[1]) * 0
        # Loop through missing features.
        for i in self._support_agent_indices[missing_indicator]:
            not_missing = self._support_agent_indices[~missing_indicator]
            support_indicies = np.append(0, not_missing)
            mean[:, i], covariance[i, i] = self._impute(i, mean, support_indicies)
        return mean, covariance


class NoImputer(Imputer):
    def __init__(self, market_data: BatchData):
        super().__init__(market_data=market_data)

    def _impute(self, i: int, mean: np.ndarray, *_):
        return mean[:, i], 0


class MeanImputer(Imputer):
    def __init__(self, market_data: BatchData):
        super().__init__(market_data=market_data)

    def _impute(self, i: int, *_):
        return np.mean(self.X_train[:, i]), 0


class RegressionImputer(Imputer):
    def __init__(self, market_data: BatchData):
        super().__init__(market_data=market_data)
        self._fitted_regression_tasks = {}
        self._prefit_regression_tasks()
        self.deterministic = False
        self.has_secondary_market = True

    def _prefit_regression_tasks(self):
        # We make the assumption that central agent features are never missing
        # and are not used the secondary market to impute missing features.
        for i in self._support_agent_indices:
            regression_task = self.regression_task()
            central_agent_indices = range(1, self.num_central_agent_features + 1)
            X = np.delete(self.X_train, [i] + list(central_agent_indices), axis=1)
            y = self.X_train[:, [i]]

            for c in chain_combinations(set(np.arange(X.shape[1])), 1, X.shape[1]):
                regression_task.fit(X, y, indices=c)
            self._fitted_regression_tasks[i] = regression_task

    def _impute(self, i: int, mean: np.ndarray, support_indicies: np.ndarray):
        regression_task = self._fitted_regression_tasks[i]
        X = mean[:, support_indicies]
        indices = np.arange(X.shape[1])
        predictive_mean, predictive_sdev = regression_task.predict(X, indices=indices)
        if self.deterministic:
            return predictive_mean, 0
        return predictive_mean, predictive_sdev

    def clear_secondary_market(
        self, x: np.ndarray, missing_indicator: np.ndarray, primary_payments: np.ndarray
    ):
        payments = np.zeros(self.X_train.shape[1])
        if np.sum(missing_indicator) == 0:  # No missing features
            return payments[1 + self.num_central_agent_features :]
        for i in self._support_agent_indices[missing_indicator]:
            payment = primary_payments[i - 1 - self.num_central_agent_features]
            not_missing = self._support_agent_indices[~missing_indicator]
            regression_task = self._fitted_regression_tasks[i]
            market_data = BatchData(
                dummy_feature=x[:, [0]],
                central_agent_features=None,
                support_agent_features=x[:, not_missing],
                target_signal=x[:, [i]],
                test_frac=1,
            )
            X_test, y_test = market_data.X_test, market_data.y_test
            attribution_policy = ShapleyAttributionPolicy(
                market_data.active_agents,
                market_data.baseline_agents,
                regression_task=regression_task,
            )

            _, allocations, _ = attribution_policy.run(X_test, y_test, payment=payment)
            for k, j in enumerate(not_missing):
                payments[j] += allocations[k] * payment
        return payments[1 + self.num_central_agent_features :]


class OlsLinearRegressionImputer(RegressionImputer):
    def __init__(self, market_data: BatchData):
        self.regression_task = MaximumLikelihoodLinearRegression
        super().__init__(market_data=market_data)
        # No uncertainty estimates provided by this imputation method.
        self.deterministic = True


class MaximumLikelihoodLinearRegressionImputer(RegressionImputer):
    def __init__(self, market_data: BatchData):
        self.regression_task = MaximumLikelihoodLinearRegression
        super().__init__(market_data=market_data)


class BayesianLinearRegressionImputer(RegressionImputer):
    def __init__(self, market_data: BatchData):
        self.regression_task = BayesianLinearRegression
        super().__init__(market_data=market_data)


class GaussianProcessLinearRegressionImputer(RegressionImputer):
    def __init__(self, market_data: BatchData):
        self.regression_task = GaussianProcessLinearRegression
        super().__init__(market_data=market_data)


def imputer_factory(market_data: BatchData, method: ImputationMethod) -> Imputer:
    """Return an Imputer based on the selected method.

    Args:
        method (ImputationMethod): imputation method.

    Returns:
        Imputer: An imputer ready to be called.
    """

    class_map = {
        ImputationMethod.no: NoImputer,
        ImputationMethod.mean: MeanImputer,
        ImputationMethod.ols: OlsLinearRegressionImputer,
        ImputationMethod.blr: BayesianLinearRegressionImputer,
        ImputationMethod.gpr: GaussianProcessLinearRegressionImputer,
        ImputationMethod.mle: MaximumLikelihoodLinearRegressionImputer,
    }
    return class_map[method](market_data=market_data)
