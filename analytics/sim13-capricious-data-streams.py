import os
from pathlib import Path
from joblib import Parallel, delayed
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats


from market.impute import imputer_factory, ImputationMethod
from market.task import GaussianProcessLinearRegression
from common.log import create_logger


if __name__ == "__main__":
    logger = create_logger(__name__)
    logger.info("Running capricious data streams analysis")

    savedir = Path(__file__).parent / "docs/sim13-capricious-data-streams"
    os.makedirs(savedir, exist_ok=True)

    sample_size = 1000
    test_frac = 0.04
    test_idx = int(sample_size * (1 - test_frac))
    noise_variance = 0.5
    num_samples = 100
    regularization = 1e-5
    coeffs = np.array([0, 0.9, 0.9, -0.9])
    num_feats = len(coeffs)

    missing_probs = np.array([0, 0, 0, 0.5])

    methods = [
        # ImputationMethod.no,
        # ImputationMethod.mean,
        ImputationMethod.blr,
        # ImputationMethod.gpr,
    ]

    losses = {
        method: np.zeros((sample_size - test_idx - 1, num_samples))
        for method in methods
    }

    for sample in tqdm(range(num_samples)):
        rho = 0.6
        X = np.random.multivariate_normal(
            [0, 0, 0], [[1, 0, 0], [0, 1, rho], [0, rho, 1]], size=sample_size
        )
        X = np.column_stack((np.ones(len(X)).reshape(-1, 1), X))
        y = (
            (X * coeffs).sum(axis=1)
            + np.random.normal(0, noise_variance**0.5, size=sample_size)
        ).reshape(-1, 1)

        X_train, X_test = X[:test_idx], X[test_idx:]
        y_train, y_test = y[:test_idx], y[test_idx:]

        regression_task = GaussianProcessLinearRegression()
        indices = np.arange(X_train.shape[1])

        regression_task.fit(X_train, y_train, indices)

        missing_indicator = (
            np.random.rand(len(X_test), len(missing_probs)) < missing_probs
        )

        imputers = [imputer_factory(X_train, y_train, method) for method in (methods)]

        for i in range(len(X_test) - 1):
            x_test = X_test[i : i + 1]
            for method, imputer in zip(methods, imputers):
                x_imputed_mean, x_imputed_covariance = imputer.impute(
                    x_test, missing_indicator[i]
                )
                # print(method, x_imputed_covariance)
                losses[method][i, sample : sample + 1] = regression_task.calculate_loss(
                    x_imputed_mean,
                    y_test[i],
                    indices,
                    X_covariance=x_imputed_covariance,
                )

    fig, ax = plt.subplots()

    for method, loss in losses.items():
        num_runs = np.arange(len(loss)) + 1
        ax.plot(loss.cumsum(axis=0).mean(axis=1) / num_runs)

    fig.savefig(savedir, dpi=300)
