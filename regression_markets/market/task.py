from typing import Tuple, Sequence
import math

import numpy as np
from scipy import stats
from scipy.stats._multivariate import multivariate_normal_frozen as mvn_frozen


class Task:
    def __init__(self, noise_variance: float = None):
        self.noise_variance = noise_variance
        self._calculate_noise_variance = (
            True if self.noise_variance is None else False
        )
        self._posteriors = {}
        self._noise_variances = {}

    def get_posterior(self, indices: Sequence):
        return self._posteriors[tuple(indices)]

    def set_posterior(self, indices: Sequence, posterior: mvn_frozen):
        self._posteriors[tuple(indices)] = posterior

    def get_noise_variance(self, indices: Sequence):
        return self._noise_variances[tuple(indices)]

    def set_noise_variance(self, indices: Sequence, noise_variance: float):
        self._noise_variances[tuple(indices)] = noise_variance

    def calculate_loss(
        self,
        X: np.ndarray,
        y: np.ndarray,
        posterior: mvn_frozen,
        noise_variance: float,
    ) -> float:
        predictive_mean, predictive_variance = self._predict(
            X, posterior, noise_variance
        )
        predictive_sdev = predictive_variance**0.5
        return -stats.norm.logpdf(
            y, loc=predictive_mean, scale=predictive_sdev
        ).mean()

    def _predictive_mean(
        self, X: np.ndarray, posterior: mvn_frozen
    ) -> np.ndarray:
        return X.dot(posterior.mean).reshape(-1, 1)

    def _predictive_variance(
        self, X: np.ndarray, posterior: mvn_frozen, noise_variance: float
    ) -> np.ndarray:
        return (
            np.einsum("ij,jk,ki->i", X, posterior.cov, X.T) + noise_variance
        ).reshape(-1, 1)

    def _predict(
        self, X: np.ndarray, posterior: mvn_frozen, noise_variance: float
    ) -> Tuple:
        predictive_mean = self._predictive_mean(X, posterior)
        predictive_variance = self._predictive_variance(
            X, posterior, noise_variance
        )
        return predictive_mean, predictive_variance


class MaximumLikelihoodLinearRegression(Task):
    def __init__(self, noise_variance: float = None):
        super().__init__(noise_variance=noise_variance)

    def _update_posterior_noise_variance(
        self, X: np.ndarray, y: np.ndarray, indices: Sequence
    ):
        if self._calculate_noise_variance:
            posterior = self.get_posterior(indices)
            predictive_mean = self._predictive_mean(X[:, indices], posterior)
            noise_variance = np.mean((predictive_mean - y) ** 2)
        else:
            noise_variance = self.noise_variance
        self.set_noise_variance(indices, noise_variance)

    def _update_posterior_coefficients(
        self, X: np.ndarray, y: np.ndarray, indices: Sequence
    ):
        # Since the posterior distribution is Gaussian, it's mode coincides with
        # its mean. Therefore by setting the posterior mean to the maximum likelihood
        # estimate and the posterior covariance to a zero matrix, we can obtain the
        # maximum likelihood predictive distribution using the same analytical method
        # of integration.
        X, y = np.atleast_2d(X)[:, indices].copy(), np.atleast_2d(y).copy()
        # print(X.shape)
        posterior_mean = (np.linalg.inv(X.T @ X) @ X.T @ y).flatten()
        posterior_covariance = np.diag(np.full(X.shape[1], math.ulp(1.0)))
        posterior = stats.multivariate_normal(
            posterior_mean, posterior_covariance
        )
        self.set_posterior(indices, posterior)

    def update_posterior(self, X: np.ndarray, y: np.ndarray, indices: Sequence):
        self._update_posterior_coefficients(X, y, indices)
        self._update_posterior_noise_variance(X, y, indices)


class BayesianLinearRegression(Task):
    def __init__(self, regularization: float, noise_variance: float = None):
        super().__init__(noise_variance=noise_variance)
        self.regularization = regularization

    def _flat_prior(self, indices):
        num_indices = len(indices)
        return stats.multivariate_normal(
            np.zeros(num_indices), np.eye(num_indices) / self.regularization
        )

    def _build_prior(self, indices: Sequence):
        if tuple(indices) in self._posteriors:
            return self.get_posterior(tuple(indices))
        return self._flat_prior(indices)

    def _update_posterior_noise_variance(
        self, X: np.ndarray, y: np.ndarray, indices: Sequence
    ):
        if self._calculate_noise_variance:
            task = MaximumLikelihoodLinearRegression(noise_variance=None)
            task.update_posterior(X, y, indices)
            noise_variance = task.get_noise_variance(indices)
        else:
            noise_variance = self.noise_variance
        self.set_noise_variance(indices, noise_variance)

    def _calculate_noise_precision(self, indices: Sequence):
        noise_variance = self.get_noise_variance(indices)
        return 1 / noise_variance

    def _update_posterior_coefficients(
        self, X: np.ndarray, y: np.ndarray, indices: Sequence
    ):

        X, y = np.atleast_2d(X)[:, indices], np.atleast_2d(y)
        noise_precision = self._calculate_noise_precision(indices)
        prior = self._build_prior(indices)
        prior_mean, prior_covariance = prior.mean.reshape(-1, 1), prior.cov
        inv_prior_covariance = np.linalg.inv(prior_covariance)
        inv_posterior_covariance = (
            inv_prior_covariance + noise_precision * X.T.dot(X)
        )
        posterior_covariance = np.linalg.inv(inv_posterior_covariance)
        posterior_mean = posterior_covariance.dot(
            inv_prior_covariance.dot(prior_mean) + noise_precision * X.T.dot(y)
        ).flatten()

        posterior = stats.multivariate_normal(
            posterior_mean, posterior_covariance
        )
        self.set_posterior(indices, posterior)

    def update_posterior(self, X: np.ndarray, y: np.ndarray, indices: Sequence):
        self._update_posterior_noise_variance(X, y, indices)
        self._update_posterior_coefficients(X, y, indices)


class OnlineBayesianLinearRegression(BayesianLinearRegression):
    def __init__(
        self,
        regularization: float,
        forgetting: float,
        noise_variance: float = None,
    ):
        super().__init__(
            regularization=regularization, noise_variance=noise_variance
        )
        self.forgetting = forgetting

    def _flatten_prior(self, indices):
        posterior = self._build_prior(indices)
        flat_prior = self._flat_prior(indices)
        posterior_covariance_inv = np.linalg.inv(posterior.cov)
        flat_prior_covariance_inv = np.linalg.inv(flat_prior.cov)
        flattened_prior_covariance_inv = (
            self.forgetting * posterior_covariance_inv
            + (1 - self.forgetting) * flat_prior_covariance_inv
        )
        flattened_prior_mean = np.linalg.inv(flattened_prior_covariance_inv) @ (
            self.forgetting * posterior_covariance_inv @ posterior.mean
            + (1 - self.forgetting)
            * flattened_prior_covariance_inv
            @ flat_prior.mean
        )
        return (
            flattened_prior_mean.reshape(-1, 1),
            flattened_prior_covariance_inv,
        )

    def _update_posterior_coefficients(
        self, X: np.ndarray, y: np.ndarray, indices: Sequence
    ):
        X, y = np.atleast_2d(X)[:, indices], np.atleast_2d(y)
        noise_precision = self._calculate_noise_precision(indices)
        (
            flattened_prior_mean,
            flattened_prior_covariance_inv,
        ) = self._flatten_prior(indices)

        posterior_cov = np.linalg.inv(
            flattened_prior_covariance_inv + noise_precision * X.T.dot(X)
        )
        posterior_mean = posterior_cov.dot(
            flattened_prior_covariance_inv.dot(flattened_prior_mean)
            + noise_precision * X.T.dot(y)
        )
        posterior = stats.multivariate_normal(
            posterior_mean.flatten(), posterior_cov
        )
        self.set_posterior(indices, posterior)
