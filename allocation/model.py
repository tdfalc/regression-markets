import numpy as np
import cvxopt as cvx
import numpy as np


class Model:
    def __init__(self):
        pass

    def calculate_weights(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def calculate_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        raise NotImplementedError


class LinearRegression(Model):
    def __init__(self):
        super().__init__()

    def calculate_weights(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.linalg.inv(X.T @ X) @ X.T @ y

    def calculate_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean((y_true - y_pred) ** 2)


class QuantileRegression(Model):
    def __init__(self, tau: float, alpha: float):
        super().__init__()
        self.tau = tau
        self.alpha = alpha

    def calculate_weights(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Quantile regression formulated as a Linear Programming problem.

        First equality constraints are added to the model, made up of positive and negative
        weights for the design matrix, positive and negative weights for the error. Since only
        positive and negative errors affect the minimzation problem, we must add a weight of 0 to
        the design matrix in the objective coeffecient vector, c. Finally, inequality constraints
        to ensure all variables are greater than 0, and the model is solved.

        Further details can be found here: https://stats.stackexchange.com/questions/384909/
            formulating-quantile-regression-as-linear-programming-problem
        """
        num_samples, num_features = X.shape
        I = np.identity(num_samples)
        A = cvx.matrix(np.concatenate((X, -X, I, -I), axis=1))

        zeros, ones = np.zeros(2 * num_features), np.ones(num_samples)
        c = cvx.matrix(np.concatenate((zeros, self.tau * ones, (1 - self.tau) * ones)))

        _, num_variables = A.size
        G = cvx.matrix(0.0, (num_variables, num_variables))
        G[:: num_variables + 1] = -1.0

        h = cvx.matrix(0.0, (num_variables, 1))

        cvx.solvers.options["glpk"] = dict(msg_lev="GLP_MSG_OFF")
        cvx.solvers.options["show_progress"] = False

        solution = cvx.solvers.lp(c, G, h, A, cvx.matrix(y), solver="glpk")["x"]

        return np.array(
            solution[0:num_features] - solution[num_features : 2 * num_features]
        ).flatten()

    def calculate_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        error = y_true - y_pred
        return np.mean(self.tau * error + self.alpha * np.log(1 + np.exp(-error / self.alpha)))
