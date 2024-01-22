from enum import Enum

import numpy as np

from common.utils import chain_combinations
from market.task import (
    MaximumLikelihoodLinearRegression,
    BayesianLinearRegression,
    GaussianProcessLinearRegression,
)


class ImputationMethod(Enum):
    no = "no"
    mean = "mean"
    ols = "ols"
    blr = "blr"
    gpr = "gpr"


class Imputer:
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y
        self._indices = np.arange(self.X.shape[1])

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
        for i in self._indices[missing_indicator]:
            not_missing = self._indices[~missing_indicator]
            mean[:, i], covariance[i, i] = self._impute(i, mean, not_missing)
        return mean, covariance


class NoImputer(Imputer):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        super().__init__(X=X, y=y)

    def _impute(self, i: int, mean: np.ndarray, *_):
        return mean[:, i], 0


class MeanImputer(Imputer):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        super().__init__(X=X, y=y)

    def _impute(self, i: int, *_):
        return np.mean(self.X[:, i]), 0


class RegressionImputer(Imputer):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        super().__init__(X=X, y=y)
        self._fitted_regression_tasks = {}
        self._prefit_regression_tasks()
        self.deterministic = False

    def _prefit_regression_tasks(self):
        for i in self._indices[1:]:  # No need to predict the dummy variable
            for c in chain_combinations(
                set(self._indices[1:]) - {i}, 1, len(self._indices) - 1
            ):
                feature_indices = (0,) + c
                X, y = self.X[:, feature_indices], self.X[:, i]
                task = self.regression_task()
                indices = np.arange(X.shape[1])
                task.fit(X, y, indices=indices)
                self._fitted_regression_tasks[(i, feature_indices)] = task

    def _impute(self, i: int, mean: np.ndarray, not_missing: np.ndarray):
        task = self._fitted_regression_tasks[(i, tuple(not_missing))]
        X = mean[:, not_missing]
        indices = np.arange(X.shape[1])
        predictive_mean, predictive_sdev = task.predict(X, indices=indices)
        if self.deterministic:
            return predictive_mean, 0
        return predictive_mean, predictive_sdev


class OlsLinearRegressionImputer(RegressionImputer):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.regression_task = MaximumLikelihoodLinearRegression
        super().__init__(X=X, y=y)
        # No uncertainty estimates provided by this imputation method.
        self.deterministic = True


class BayesianLinearRegressionImputer(RegressionImputer):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.regression_task = BayesianLinearRegression
        super().__init__(X=X, y=y)


class GaussianProcessLinearRegressionImputer(RegressionImputer):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.regression_task = GaussianProcessLinearRegression
        super().__init__(X=X, y=y)


def imputer_factory(X: np.ndarray, y: np.ndarray, method: ImputationMethod) -> Imputer:
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
    }
    return class_map[method](X, y)
