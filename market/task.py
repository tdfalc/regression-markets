from typing import Tuple, Sequence, Any
import math
import warnings

import numpy as np
from scipy import stats
from scipy.stats._multivariate import multivariate_normal_frozen as mvn_frozen
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, DotProduct
from sklearn.exceptions import ConvergenceWarning


## TODO: Remove requirement to pass both X and indices when calling fit and predict.
class Task:
    def __init__(self):
        self._noise_variances = {}

    def fit(self, X: np.ndarray, y: np.ndarray, indices: Sequence):
        raise NotImplementedError

    def predict(self, X: np.ndarray, indices: np.ndarray, **_) -> Tuple:
        """Calculate predictive mean and variance for query points.

        NOTE: Indicies provided in addition to query points to provide a look up
        NOTE: to the correct posterior.
        """
        raise NotImplementedError

    def calculate_loss(
        self, X: np.ndarray, y: np.ndarray, indices: np.ndarray
    ) -> float:
        raise NotImplementedError

    def get_posterior_noise_variance(self, indices: Sequence):
        return self._noise_variances[tuple(indices)]

    def set_posterior_noise_variance(self, indices: Sequence, noise_variance: Any):
        self._noise_variances[tuple(indices)] = noise_variance

    def calculate_loss(
        self, X: np.ndarray, y: np.ndarray, indices: np.ndarray, X_covariance
    ) -> float:
        predictive_mean, predictive_sdev = self.predict(
            X=X, indices=indices, X_covariance=X_covariance
        )
        return -stats.norm.logpdf(y, loc=predictive_mean, scale=predictive_sdev).mean()


class WeightSpaceTask(Task):
    def __init__(self):
        super().__init__()
        self._posteriors = {}

    def get_posterior_coefficients(self, indices: Sequence):
        return self._posteriors[tuple(indices)]

    def set_posterior_coefficients(self, indices: Sequence, posterior: mvn_frozen):
        self._posteriors[tuple(indices)] = posterior

    def fit(self, X: np.ndarray, y: np.ndarray, indices: Sequence):
        self._update_posterior(X, y, indices)

    def _update_posterior(self, X: np.ndarray, y: np.ndarray, indices: Sequence):
        raise NotImplementedError

    def _predictive_mean(self, X: np.ndarray, coefficients: mvn_frozen):
        return X.dot(coefficients.mean).reshape(-1, 1)


class MaximumLikelihoodLinearRegression(WeightSpaceTask):
    def __init__(self):
        super().__init__()

    @classmethod
    def _ols_solution(self, X: np.ndarray, y: np.ndarray):
        return np.linalg.inv(X.T @ X) @ X.T @ y

    def _update_posterior(self, X: np.ndarray, y: np.ndarray, indices: Sequence):
        # Since the posterior distribution is Gaussian, it's mode coincides with
        # its mean. Therefore by setting the posterior mean to the maximum likelihood
        # estimate and the posterior covariance to a zero matrix, we can obtain the
        # maximum likelihood predictive distribution using the same analytical method
        # of integration.
        X, y = np.atleast_2d(X)[:, indices].copy(), y.reshape(-1, 1).copy()
        posterior_mean = self._ols_solution(X, y).flatten()
        posterior_covariance = np.diag(np.full(X.shape[1], math.ulp(1.0)))
        posterior_coefficients = stats.multivariate_normal(
            posterior_mean, posterior_covariance
        )
        self.set_posterior_coefficients(indices, posterior_coefficients)

        # Calculate maximum likelihood estimate for the noise variance.
        predictive_mean = self._predictive_mean(X, posterior_coefficients)
        noise_variance = np.mean((predictive_mean - y) ** 2)
        self.set_posterior_noise_variance(indices, noise_variance)

    def predict(self, X: np.ndarray, indices: np.ndarray, **_) -> Tuple:
        posterior_coefficients = self.get_posterior_coefficients(indices)
        noise_variance = self.get_posterior_noise_variance(indices)
        predictive_mean = self._predictive_mean(X, posterior_coefficients)
        coefficient_variance = np.einsum(
            "ij,jk,ki->i", X, posterior_coefficients.cov, X.T
        )
        predictive_variance = (coefficient_variance + noise_variance).reshape(-1, 1)
        return predictive_mean, np.sqrt(predictive_variance)


class BayesianLinearRegression(WeightSpaceTask):
    """Bayesian linear regression with Normal-Inverse-Gamma prior."""

    def __init__(self):
        super().__init__()

    def _update_posterior(self, X: np.ndarray, y: np.ndarray, indices: Sequence):
        X, y = np.atleast_2d(X)[:, indices], y.reshape(-1, 1).copy()
        prior_coefficients = self._build_prior_coefficients(indices)
        prior_mean = prior_coefficients.mean.reshape(-1, 1)
        prior_covariance = prior_coefficients.cov

        # Calculate parameters of multivariate Gaussian posterior coefficients.
        prior_precision = np.linalg.inv(prior_covariance)
        posterior_precision = prior_precision + X.T.dot(X)
        posterior_covariance = np.linalg.inv(posterior_precision)
        posterior_mean = posterior_covariance.dot(
            prior_precision.dot(prior_mean) + X.T.dot(y)
        )
        posterior_coefficients = stats.multivariate_normal(
            posterior_mean.flatten(), posterior_covariance
        )
        self.set_posterior_coefficients(indices, posterior_coefficients)

        # Calculate parameters of Inverse-Gamma posterior noise variance.
        posterior_a = 1e-9 + len(X) / 2
        posterior_b = 1e-9 + 0.5 * (
            y.T @ y
            + prior_mean.T @ prior_precision @ prior_mean
            - posterior_mean.T @ posterior_precision @ posterior_mean
        )
        posterior_noise_variance = stats.invgamma(a=posterior_a, scale=posterior_b)
        self.set_posterior_noise_variance(indices, posterior_noise_variance)

    def _flat_prior_coefficients(self, indices):
        num_indices = len(indices)
        return stats.multivariate_normal(
            np.zeros(num_indices), np.eye(num_indices) / 1e-5  # Uninformative prior
        )

    def _build_prior_coefficients(self, indices: Sequence):
        if tuple(indices) in self._posteriors:
            return self.get_posterior_coefficients(tuple(indices))
        return self._flat_prior_coefficients(indices)

    def predict(self, X: np.ndarray, indices: np.ndarray, **_) -> Tuple:
        coefficients = self.get_posterior_coefficients(indices)
        noise_variance = self.get_posterior_noise_variance(indices)
        predictive_mean = self._predictive_mean(X, coefficients)
        a, b = noise_variance.kwds["a"], noise_variance.kwds["scale"]
        scale = (b / a) * (X @ coefficients.cov @ X.T + np.eye(len(X)))
        dof = 2 * a
        predictive_variance = np.diag((dof / (dof - 2)) * scale)
        return predictive_mean, np.sqrt(predictive_variance)


class GaussianProcessLinearRegression(Task):
    def __init__(self):
        super().__init__()
        self._observations = {}

    def _amplitude(self, X: np.ndarray):
        return np.eye(X.shape[1]) / 1e-5

    def _kernel_fn(self, X: np.ndarray, Z: np.ndarray):
        return X @ self._amplitude(X) @ Z.T

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        indices: Sequence,
        n_restarts_optimizer: int = 0,
    ):
        kernel = DotProduct(sigma_0_bounds=(1e-10, 1e-9)) + WhiteKernel()
        gp = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=n_restarts_optimizer
        )
        with warnings.catch_warnings():
            # Ignore bounds warning for regularization parameter
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            gp.fit(X[:, indices], y)
        noise_variance = gp.kernel_.get_params()["k2__noise_level"]
        self.set_posterior_noise_variance(indices, noise_variance)
        self._observations[tuple(indices)] = (X[:, indices], y)

    def _input_noise_free_posterior(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        query_mean: np.ndarray,
        noise_variance: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Kernel of observations
        kernel_train = self._kernel_fn(X_train, X_train)
        kernel_train += noise_variance * np.eye(len(X_train))
        # Kernel of observations vs. query points
        kernel_train_test = self._kernel_fn(X_train, query_mean)
        # Kernel of query points
        kernel_test = self._kernel_fn(query_mean, query_mean)
        kernel_test += noise_variance * np.eye(len(query_mean))
        # Compute posterior
        solved = (np.linalg.inv(kernel_train) @ kernel_train_test).T
        mean = solved @ y_train
        covariance = kernel_test - solved @ kernel_train_test
        # For now we are only interested in the diagonal
        sdev = np.sqrt(covariance.diagonal().reshape(-1, 1))
        return mean, sdev

    def predict(
        self,
        X: np.ndarray,
        indices: np.ndarray,
        X_covariance: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Gaussian Process posterior moments with possibly noisy query points.

        Args:
            X (np.ndarray): the array of index point means at which the resulting posterior predictive
                distribution over function values is defined.
            indices (np.ndarray): array of indicies to look up posterior noise variance.
            X_covariance (np.ndarray): the corresponding array of index point covariances. The shape of
                this array will be (N, M, M), where M is the number of features and N is the number of
                query points.

        Returns:
            Tuple[np.ndarray, np.ndarray]: mean and standard deviation of Gaussian posterior
                predictive distribution.
        """
        X_train, y_train = self._observations[tuple(indices)]
        noise_variance = self.get_posterior_noise_variance(indices)
        mean, sdev = self._input_noise_free_posterior(
            X_train, y_train, X, noise_variance
        )
        ## TODO: Break out this job from this function.
        X_covariance = X_covariance[:, indices][indices, :].copy()
        if X_covariance is None or np.count_nonzero(X_covariance) == 0:
            # If no covariance is provided, the noise-free posterior is returned
            # to avoid unecessary computations (i.e., we would get the same result
            # assuming the covariance was a zero matrix)
            return mean, sdev

        num_features = X_train.shape[1]
        X_covariance = (
            np.atleast_3d(X_covariance).reshape(-1, num_features, num_features).copy()
        )

        # Kernel of the observations
        kernel_train = self._kernel_fn(X_train, X_train)
        kernel_train += noise_variance * np.eye(len(X_train))
        kernel_train_inv = np.linalg.inv(kernel_train)
        # Extract the weights applied to the kernel of the query points when computing
        # the posterior predictive mean
        weights = kernel_train_inv @ y_train
        # We can decompose the variance into that given by the noise-free posterior
        variance = np.square(sdev)
        # With additional correction terms
        variance += np.trace(
            self._amplitude(X_train) @ X_covariance, axis1=1, axis2=2
        ).reshape(-1, 1)
        variance -= np.trace(
            self._amplitude(X_train)
            @ X_train.T
            @ (kernel_train_inv - np.outer(weights, weights))
            @ X_train
            @ self._amplitude(X_train)
            @ X_covariance,
            axis1=1,
            axis2=2,
        ).reshape(-1, 1)
        return mean, np.sqrt(variance)
