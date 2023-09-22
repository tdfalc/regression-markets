import itertools
from typing import Tuple

import numpy as np


class DataGenerator:
    def __init__(
        self,
        variables: np.ndarray,
        additive_noise: np.ndarray,
        multiplicative_noise: np.ndarray,
    ):
        """Instantiate class to generate synthetic data.

        Args:
            variables (np.ndarray): Array of raw feature variables.
            additive_noise (float): The additive gaussian noise added to the target variable.
            multiplicative_noise (Sequence[float]): The multiplicative gaussian noise added
                to each input features.
        """
        self.variables = variables
        self.additive_noise = additive_noise
        self.multiplicative_noise = multiplicative_noise

    def _build_design_matrix(self, polynomial_order: int):
        X = self.variables
        for i in range(1, polynomial_order):
            indicies = np.arange(self.variables.shape[1])
            combinations = list(itertools.combinations_with_replacement(indicies, i + 1))
            X = np.hstack([X] + [X[:, c].prod(axis=1).reshape(-1, 1) for c in combinations])
        return np.insert(X, 0, 1, axis=1)

    def linear_transformation(
        self, weights: np.ndarray, polynomial_order: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """The target is modelled as a linear transformation (w.r.t the weights) of the 
        feature variables,with polynomial terms up to the specified order. 
        
        The polynomial combinations are emitted in lexicographic ordering, therefore weights 
        should be ordered accordingly. For example, for two variables, the second order polynomial 
        model of the target variable is of the form:

            y = w_{0} + w_{1] * x_{1} + w_{2} * x_{2} + w_{3} * x_{3} + w_{4} * x_{1}^{2} + \\
                w_{5} * x_{1} * x_{2} + w_{6} * x_{2}^{2} + ϵ
        """
        X = self._build_design_matrix(polynomial_order)
        y = np.sum((X * self.multiplicative_noise) * weights, axis=1)
        return X, y

    def linear_transformation_with_lagged_target(
        self, weights: np.ndarray, polynomial_order: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """The target is modelled as a linear transformation (w.r.t the weights) of the
        feature variables, with polynomial terms up to the specified order, plus an autoregressive
        term.

        The polynomial combinations are emitted in lexicographic ordering, therefore weights should
        be ordered accordingly. The autoregressive term is added as column 2 in design matrix (i.e.)
        right-adjacent to the dummy variable. For example, for two variables, the first order model
        of the target variable is of the form:

            y = w_{0} + w_{1} * y_{t-1} + w{2} * x_{3} + w_{4} * x_{2} + w_{5} * x_{3}
        """
        X = self._build_design_matrix(polynomial_order)
        y = np.zeros(len(X))
        weights = np.broadcast_to(weights, (X.shape[0], X.shape[1] + 1))
        for i in range(1, len(X)):
            y[i] = weights[i, 1:] @ (X[i] * self.multiplicative_noise[i, :]).T
            y[i] += weights[i, 1] * y[i - 1]
            y[i] += self.additive_noise[i]

        X = np.insert(X, 1, np.pad(y, 1)[:-2], axis=1)

        return X, y
