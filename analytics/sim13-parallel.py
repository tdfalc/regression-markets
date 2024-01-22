import os
from pathlib import Path
from joblib import Parallel, delayed
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats


from market.impute import imputer_factory, ImputationMethod
from market.task import GaussianProcessLinearRegression, BayesianLinearRegression
from market.data import BatchData
from market.policy import ShapleyAttributionPolicy
from common.log import create_logger
from common.utils import tqdm_joblib, chain_combinations

if __name__ == "__main__":
    logger = create_logger(__name__)
    logger.info("Running capricious data streams analysis")

    savedir = Path(__file__).parent / "docs/sim13-capricious-data-streams"
    os.makedirs(savedir, exist_ok=True)

    sample_size = 5000
    test_frac = 0.04
    test_idx = int(sample_size * (1 - test_frac))
    noise_variance = 0.5
    num_samples = 500
    regularization = 1e-5
    coeffs = np.array([0, 0.9, 0.9, -0.9])
    num_feats = len(coeffs)
    payment = 1

    missing_probs = np.array([0, 0, 0, 0.5])

    methods = [
        ImputationMethod.no,
        ImputationMethod.mean,
        ImputationMethod.ols,
        # ImputationMethod.blr,
        # ImputationMethod.gpr,
    ]

    def _run_experiment():
        rho = 0.6
        X = np.random.multivariate_normal(
            [0, 0, 0], [[1, 0, 0], [0, 1, rho], [0, rho, 1]], size=sample_size
        )
        X = np.column_stack((np.ones(len(X)).reshape(-1, 1), X))
        y = (
            (X * coeffs).sum(axis=1)
            + np.random.normal(0, noise_variance**0.5, size=sample_size)
        ).reshape(-1, 1)

        market_data = BatchData(
            dummy_feature=X[:, [0]],
            central_agent_features=X[:, [1]],
            support_agent_features=X[:, 2:],
            target_signal=y,
            test_frac=test_frac,
        )

        X_train, X_test = market_data.X_train, market_data.X_test
        y_train, y_test = market_data.y_train, market_data.y_test

        regression_task = BayesianLinearRegression()  # needs to be gpr
        indices = np.arange(X_train.shape[1])

        num_features = X.shape[1]
        for indices in chain_combinations(np.arange(num_features), 1, num_features):
            regression_task.fit(X, y, indices)

        missing_indicator = (
            np.random.rand(len(X_test), len(missing_probs)) < missing_probs
        )

        imputers = [imputer_factory(X_train, y_train, method) for method in (methods)]

        losses = {
            method: np.zeros((sample_size - test_idx - 1, 1)) for method in methods
        }

        market_outcomes = {
            "contributions": {
                method: np.zeros(
                    (sample_size - test_idx - 1, market_data.num_support_agent_features)
                )
                for method in methods
            },
            "allocations": {
                method: np.zeros(
                    (sample_size - test_idx - 1, market_data.num_support_agent_features)
                )
                for method in methods
            },
            "payments": {
                method: np.zeros(
                    (sample_size - test_idx - 1, market_data.num_support_agent_features)
                )
                for method in methods
            },
        }

        for i in range(len(X_test) - 1):
            x_test = X_test[i : i + 1]
            for method, imputer in zip(methods, imputers):
                x_imputed_mean, x_imputed_covariance = imputer.impute(
                    x_test, missing_indicator[i]
                )
                losses[method][i, :] = regression_task.calculate_loss(
                    x_imputed_mean,
                    y_test[i],
                    indices,
                    X_covariance=x_imputed_covariance,
                )

                attribution_policy = ShapleyAttributionPolicy(
                    active_agents=market_data.active_agents,
                    baseline_agents=market_data.baseline_agents,
                    regression_task=regression_task,
                )

                (
                    market_outcomes["contributions"][method][i, :],
                    market_outcomes["allocations"][method][i, :],
                    market_outcomes["payments"][method][i, :],
                ) = attribution_policy.run(
                    x_imputed_mean,
                    y_test[i],
                    X_covariance=x_imputed_covariance,
                    payment=payment,
                )

        return losses, market_outcomes

    with tqdm_joblib(
        tqdm(desc="Simulations", total=num_samples, position=2, leave=False)
    ) as _:
        results = Parallel(n_jobs=-1)(
            delayed(_run_experiment)() for _ in range(num_samples)
        )

        losses, market_outcomes = list(zip(*results))

    fig, ax = plt.subplots()

    for method in methods:
        loss = np.stack([l[method] for l in losses])
        num_runs = (np.arange(loss.shape[1]) + 1).reshape(-1, 1)
        ax.plot(loss.cumsum(axis=1).mean(axis=0) / num_runs, label=method)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Expected Loss")
    ax.legend()

    fig.savefig(savedir / "losses", dpi=300)

    fig, ax = plt.subplots()

    metric = "payments"
    for i, method in enumerate(methods):
        payments = np.stack([o[metric][method] for o in market_outcomes])
        num_runs = 1  # np.arange(payments.shape[1]) + 1
        for j in (0, 1):
            ax.plot(
                payments.cumsum(axis=1).mean(axis=0)[:, j] / num_runs,
                label=method if j == 0 else "",
                color=f"C{i}",
                ls="solid" if j == 0 else "dashed",
            )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Expected Payment")
    ax.legend()

    fig.savefig(savedir / "payments", dpi=300)
