from typing import Tuple, Sequence
import math

import numpy as np
from scipy import stats
from scipy.stats._multivariate import multivariate_normal_frozen as mvn_frozen


class Task:
    def __init__(self, noise_variance: float = None):
        self.noise_variance = noise_variance
        self._calculate_noise_variance = True if self.noise_variance is None else False

    def fit(self):
        raise NotImplementedError

    def predict(self) -> Tuple:
        raise NotImplementedError

    def calculate_loss(self) -> float:
        raise NotImplementedError


class WeightSpaceTask(Task):
    def __init__(self, noise_variance: float = None):
        super().__init__(noise_variance=noise_variance)
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

    def fit(self, X: np.ndarray, y: np.ndarray, indices: Sequence):
        self._update_posterior_coefficients(X, y, indices)
        self._update_posterior_noise_variance(X, y, indices)

    def _update_posterior_coefficients(self):
        raise NotImplementedError

    def _update_posterior_noise_variance(self):
        raise NotImplementedError

    @classmethod
    def _posterior_mean(self, X: np.ndarray, y: np.ndarray):
        return np.linalg.inv(X.T @ X) @ X.T @ y

    def predict(self, X: np.ndarray, indices: np.ndarray) -> Tuple:
        posterior = self.get_posterior(indices)
        noise_variance = self.get_noise_variance(indices)
        predictive_mean = X.dot(posterior.mean).reshape(-1, 1)
        coefficient_variance = np.einsum("ij,jk,ki->i", X, posterior.cov, X.T)
        predictive_variance = (coefficient_variance + noise_variance).reshape(-1, 1)
        return predictive_mean, predictive_variance

    def calculate_loss(
        self, X: np.ndarray, y: np.ndarray, indices: np.ndarray
    ) -> float:
        predictive_mean, predictive_variance = self.predict(X, indices)
        predictive_sdev = np.sqrt(predictive_variance)
        return -stats.norm.logpdf(y, loc=predictive_mean, scale=predictive_sdev).mean()


class MaximumLikelihoodLinearRegression(WeightSpaceTask):
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
        posterior_mean = self._posterior_mean(X, y).flatten()
        posterior_covariance = np.diag(np.full(X.shape[1], math.ulp(1.0)))
        posterior = stats.multivariate_normal(posterior_mean, posterior_covariance)
        self.set_posterior(indices, posterior)


class BayesianLinearRegression(WeightSpaceTask):
    def __init__(self, regularization: float, noise_variance: float = None):
        super().__init__(noise_variance=noise_variance)
        self.regularization = regularization

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
        inv_posterior_covariance = inv_prior_covariance + noise_precision * X.T.dot(X)
        posterior_covariance = np.linalg.inv(inv_posterior_covariance)
        posterior_mean = posterior_covariance.dot(
            inv_prior_covariance.dot(prior_mean) + noise_precision * X.T.dot(y)
        ).flatten()
        posterior = stats.multivariate_normal(posterior_mean, posterior_covariance)
        self.set_posterior(indices, posterior)

    def _flat_prior(self, indices):
        num_indices = len(indices)
        return stats.multivariate_normal(
            np.zeros(num_indices), np.eye(num_indices) / self.regularization
        )

    def _build_prior(self, indices: Sequence):
        if tuple(indices) in self._posteriors:
            return self.get_posterior(tuple(indices))
        return self._flat_prior(indices)


class FunctionSpaceTask(Task):
    def __init__(self, noise_variance: float = None):
        super().__init__(noise_variance=noise_variance)

    def _kernel_fn(self):
        raise NotImplementedError


# class LinearGPRegression(Task):
#     def __init__(self, noise_variance: float = None):
#         super().__init__(noise_variance=noise_variance)

#     def _amplitude(self, X: np.ndarray):
#         return np.eye(X.shape[1]) / 1e-5

#     def _kernel_fn(self, X: np.ndarray, Z: np.ndarray):
#         return X @ self._amplitude(X) @ Z.T

#     def _update_posterior_noise_variance(
#         self, X: np.ndarray, y: np.ndarray, indices: Sequence
#     ):
#         if self._calculate_noise_variance:
#             task = MaximumLikelihoodLinearRegression(noise_variance=None)
#             task.update_posterior(X, y, indices)
#             noise_variance = task.get_noise_variance(indices)
#         else:
#             noise_variance = self.noise_variance
#         self.set_noise_variance(indices, noise_variance)

#     def _posterior_noise_precision(self, X: np.ndarray, y: np.ndarray):
#         if self.noise_precision is None:
#             return mle_precision(X, y)
#         return self.noise_precision

#     def _noise_free_posterior(
#         self, query_mean: np.ndarray, X_obs: np.ndarray, y_obs: np.ndarray
#     ) -> Tuple[np.ndarray, np.ndarray]:
#         noise_variance = 1 / self._posterior_noise_precision(X_obs, y_obs)
#         # Kernel of the observations
#         kernel_train = self.kernel_fn(X_obs, X_obs)
#         kernel_train += noise_variance * np.eye(len(X_obs))
#         # Kernel of observations vs. query points
#         kernel_train_test = self.kernel_fn(X_obs, query_mean)
#         # Kernel of query points
#         kernel_test = self.kernel_fn(query_mean, query_mean)
#         kernel_test += noise_variance * np.eye(len(query_mean))
#         # Compute posterior
#         solved = (np.linalg.inv(kernel_train) @ kernel_train_test).T
#         mean = solved @ y_obs
#         covariance = kernel_test - solved @ kernel_train_test
#         # For now we are only interested in the diagonal
#         sdev = np.sqrt(covariance.diagonal().reshape(-1, 1))
#         return mean, sdev

#     def posterior(
#         self,
#         X_obs: np.ndarray,
#         y_obs: np.ndarray,
#         query_mean: np.ndarray,
#         query_covariance: np.ndarray = None,
#     ) -> Tuple[np.ndarray, np.ndarray]:
#         """Compute Gaussian Process posterior moments with possibly noisy query points.

#         Args:
#             X_obs (np.ndarray): the array of index points corresponding to the observations.
#             y_obs (np.ndarray): the array of (noisy) observations.
#             query_mean (np.ndarray): the array of index point means at which the resulting posterior
#                 predictive distribution over function values is defined.
#             query_covariance (np.ndarray): the corresponding array of index point covariances. The
#                 shape of this array will be (M, M, N), where M is the number of features and N is
#                 the number of query points.
#         Returns:
#             Tuple[np.ndarray, np.ndarray]: mean and standard deviation of Gaussian posterior
#                 predictive distribution.
#         """
#         noise_variance = 1 / self._posterior_noise_precision(X_obs, y_obs)
#         mean, sdev = self._noise_free_posterior(query_mean, X_obs, y_obs)
#         if query_covariance is None:
#             # If no covariance is provided, the noise-free posterior is returned
#             # to avoid unecessary computations (i.e., we would get the same result
#             # assuming the covariance was a zero matrix)
#             return mean, sdev
#         # Kernel of the observations
#         kernel_train = self.kernel_fn(X_obs, X_obs)
#         kernel_train += noise_variance * np.eye(len(X_obs))
#         kernel_train_inv = np.linalg.inv(kernel_train)
#         # Extract the weights applied to the kernel of the query points when computing
#         # the posterior predictive mean
#         weights = kernel_train_inv @ y_obs
#         # We can decompose the variance into that given by the noise-free posterior
#         variance = np.square(sdev)
#         # With additional correction terms
#         variance += np.trace(self._amplitude(X_obs) @ query_covariance)
#         variance -= np.trace(
#             self._amplitude(X_obs)
#             @ X_obs.T
#             @ (kernel_train_inv - np.outer(weights, weights))
#             @ X_obs
#             @ self._amplitude(X_obs)
#             @ query_covariance
#         )

#         return mean, np.sqrt(variance)
