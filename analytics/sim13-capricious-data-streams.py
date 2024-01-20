import os
from pathlib import Path
from joblib import Parallel, delayed
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats


from market.impute import MeanImputer, LinearRegressionImputer
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

    loss_orig = np.zeros((sample_size - test_idx - 1, num_samples))
    loss_meanimpute = np.zeros((sample_size - test_idx - 1, num_samples))
    loss_linregimpute = np.zeros((sample_size - test_idx - 1, num_samples))
    loss_gp = np.zeros((sample_size - test_idx - 1, num_samples))

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

        prior_mean = np.zeros(X.shape[1])
        prior_covariance = np.eye(X.shape[1]) / regularization
        posterior_covariance = np.linalg.inv(
            (1 / noise_variance) * X_train.T @ X_train + np.linalg.inv(prior_covariance)
        )
        posterior_mean = posterior_covariance @ (
            (1 / noise_variance) * X_train.T @ y_train
            + np.linalg.inv(prior_covariance) @ prior_mean.reshape(-1, 1)
        )

        missing_indicator = (
            np.random.rand(len(X_test), len(missing_probs)) < missing_probs
        )

        mean_imputer = MeanImputer(X_train, y_train)
        linreg_imputer = LinearRegressionImputer(X_train, y_train)

        for i in range(len(X_test) - 1):
            # Assume no missing valuez
            x_test = X_test[i : i + 1]
            predictive_mean = (x_test @ posterior_mean).ravel()
            predictive_sdev = (
                x_test @ posterior_covariance @ x_test.T + noise_variance
            ).ravel() ** 0.5
            loss_orig[i, sample : sample + 1] = (
                -stats.norm(
                    loc=predictive_mean,
                    scale=predictive_sdev,
                )
                .logpdf(y_test[i])
                .ravel()
            )

            # Mean imputation
            x_test_mean_impute, _ = mean_imputer.impute(x_test, missing_indicator[i])
            predictive_mean = (x_test_mean_impute @ posterior_mean).ravel()
            predictive_sdev = (
                x_test_mean_impute @ posterior_covariance @ x_test_mean_impute.T
                + noise_variance
            ).ravel() ** 0.5
            loss_meanimpute[i, sample : sample + 1] = (
                -stats.norm(loc=predictive_mean, scale=predictive_sdev)
                .logpdf(y_test[i])
                .ravel()
            )

            # Linear regression imputation
            x_test_linregimpute, _ = linreg_imputer.impute(x_test, missing_indicator[i])
            predictive_mean = (x_test_linregimpute @ posterior_mean).ravel()
            predictive_sdev = (
                x_test_linregimpute @ posterior_covariance @ x_test_linregimpute.T
                + noise_variance
            ).ravel() ** 0.5
            loss_linregimpute[i, sample : sample + 1] = (
                -stats.norm(loc=predictive_mean, scale=predictive_sdev)
                .logpdf(y_test[i])
                .ravel()
            )

            # Gaussian process imputation
            # # GP regression imputation
            u_mean, u_cov = gp_imputer.impute(x_test, missing_indicator[i])
            gp = LinearGP(1 / noise_variance)
            pred_mean, pred_sdev = gp.posterior(
                X_train,
                y_train,
                u_mean,
                query_covariance=np.expand_dims(u_cov, axis=-1),
            )
            loss_gp[i, sample : sample + 1] = (
                -stats.norm(
                    loc=pred_mean.ravel()[0],
                    scale=pred_sdev.ravel()[0],
                )
                .logpdf(y_test[i])
                .ravel()
            )

    fig, ax = plt.subplots()
    num_runs = np.arange(len(loss_linregimpute)) + 1
    ax.plot(loss_orig.cumsum(axis=0).mean(axis=1) / num_runs)
    ax.plot(loss_meanimpute.cumsum(axis=0).mean(axis=1) / num_runs)
    ax.plot(loss_linregimpute.cumsum(axis=0).mean(axis=1) / num_runs)
    ax.plot(loss_gp.cumsum(axis=0).mean(axis=1) / num_runs)
    fig.savefig(savedir, dpi=300)
