import numpy as np


class StateTracker:
    def __init__(
        self,
        forgetting_factor: float,
        initial_weights: np.ndarray,
        contributions_burn_in: int,
        weights_burn_in: int,
    ):
        self.forgetting_factor = forgetting_factor
        self.initial_weights = initial_weights
        self.contributions_burn_in = contributions_burn_in
        self.weights_burn_in = weights_burn_in
        self._estimated_weights = np.tile(initial_weights, (self.weights_burn_in, 1))
        self._memory = None

    @property
    def current_weights(self):
        return self._estimated_weights[-1, :]

    def log_data(self, X: np.ndarray, y: np.ndarray):
        self.X, self.y = X, y
        self.error = self.y - np.sum(self._estimated_weights[-1, :] * self.X)

    def _psd_matrix(self):
        return self.X.reshape(-1, 1) @ self.X.reshape(-1, 1).T

    def update_weights(self):
        raise NotImplementedError

    def _update_memory(self):
        raise NotImplementedError

    def _log_updated_weights(self, weights: np.ndarray):
        self._estimated_weights = np.insert(
            self._estimated_weights, len(self._estimated_weights), weights, axis=0
        )

    def update_memory(self):
        _, num_features = self.X.reshape(1, -1).shape
        if self._memory is None:
            self._memory = np.zeros((num_features, num_features))
        self._update_memory()


class QuadraticStateTracker(StateTracker):
    def __init__(
        self,
        forgetting_factor: float,
        initial_weights: np.ndarray,
        contributions_burn_in: int,
        weights_burn_in: int,
    ):
        super().__init__(forgetting_factor, initial_weights, contributions_burn_in, weights_burn_in)

    def _update_memory(self):
        self._memory = self._memory * self.forgetting_factor + self._psd_matrix()

    def update_weights(self):
        updated_weights = self.current_weights + np.linalg.solve(
            self._memory, self.error * self.X.flatten()
        )
        self._log_updated_weights(updated_weights)


class QuantileStateTracker(StateTracker):
    def __init__(
        self,
        forgetting_factor: float,
        initial_weights: np.ndarray,
        contributions_burn_in: int,
        weights_burn_in: int,
        alpha: float,
        tau: float,
    ):
        super().__init__(forgetting_factor, initial_weights, contributions_burn_in, weights_burn_in)
        self.alpha = alpha
        self.tau = tau

    def _update_memory(self):
        self._memory = self._memory * self.forgetting_factor + (
            (1 - self.forgetting_factor)
            * self._psd_matrix()
            * np.exp(-self.error / self.alpha)
            * (1 + self.alpha)
            / (1 + np.exp(-self.error / self.alpha)) ** 2
        )

    def update_weights(self):
        updated_weights = self.current_weights - (1 - self.forgetting_factor) * np.linalg.solve(
            self._memory,
            -self.X.flatten()
            * (
                self.tau
                + self.alpha
                * (1 - 1 / self.alpha * np.exp(-self.error / self.alpha))
                / (1 + np.exp(-self.error / self.alpha))
            ),
        )
        self._log_updated_weights(updated_weights)
