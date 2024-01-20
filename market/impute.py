from enum import Enum

import numpy as np

from common.utils import chain_combinations
from market.task import MaximumLikelihoodLinearRegression as MLE, LinearGPRegression


class ImputationMethod(Enum):
    mean = "mean"
    ols = "ols"
    gp = "gp"


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


class MeanImputer(Imputer):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        super().__init__(X=X, y=y)

    def _impute(self, i: int, *_):
        return np.mean(self.X[:, i]), 0


class LinearRegressionImputer(Imputer):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        super().__init__(X=X, y=y)
        self._coefficient_estimates = {}
        self._precalculate_coefficient_estimates()

    def _precalculate_coefficient_estimates(self):
        for i in self._indices[1:]:  # No need to predict the dummy variable
            for c in chain_combinations(
                set(self._indices[1:]) - {i}, 1, len(self._indices) - 1
            ):
                feature_indices = (0,) + c
                X, y = self.X[:, feature_indices], self.X[:, i]
                self._coefficient_estimates[(i, feature_indices)] = MLE._posterior_mean(
                    X, y
                )

    def _impute(self, i: int, mean: np.ndarray, not_missing: np.ndarray):
        coefficient_estimates = self._coefficient_estimates[(i, tuple(not_missing))]
        # No uncertainty estimates provided by this imputation method.
        return mean[:, not_missing] @ coefficient_estimates, 0


class GaussianProcessImputer:
    def __init__(self, X: np.ndarray, y: np.ndarray):
        super().__init__(X=X, y=y)
        self._coefficient_estimates = {}
        self._precalculate_coefficient_estimates()

    def _precalculate_coefficient_estimates(self):
        indices = np.arange(X.shape[1])
        for i in indices[1:]:  # No need to predict the dummy variable
            for c in chain_combinations(set(indices[1:]) - {i}, 1, len(indices) - 1):
                feature_indices = (0,) + c
                X, y = self.X[:, feature_indices], self.X[:, i]
                self._coefficient_estimates[(i, c)] = ols_solution(X, y)

    def _impute(self, i: int, mean: np.ndarray, not_missing: np.ndarray):
        noise_precision = self._noise_precisions[(i, tuple(not_missing))]
        gp = LinearGP(noise_precision)
        mean, sdev = gp.posterior(
            self.X[:, not_missing], self.X[:, i], mean[:, not_missing]
        )
        return mean, np.square(sdev)
