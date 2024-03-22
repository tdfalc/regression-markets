import numpy as np
import pandas as pd
from pathlib import Path
import os

import os
from pathlib import Path
from joblib import Parallel, delayed
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats


import os
from pathlib import Path
from joblib import Parallel, delayed
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats


from market.impute import imputer_factory, ImputationMethod, RegressionImputer
from market.task import GaussianProcessLinearRegression, BayesianLinearRegression
from market.data import BatchData
from market.mechanism import Market
from market.policy import ShapleyAttributionPolicy
from common.log import create_logger
from common.utils import tqdm_joblib, cache


from market.impute import imputer_factory, ImputationMethod
from market.task import GaussianProcessLinearRegression
from common.log import create_logger


def parse_raw_data(fname: Path) -> pd.DataFrame:
    df = pd.read_csv(fname, header=None, index_col=0)
    df.index = pd.to_datetime(df.index, format="%d/%m/%Y %H.%M")
    df = df.tz_localize("utc")
    df.columns = [f"a{i}" for i in df.columns]
    return df


if __name__ == "__main__":
    savedir = Path(__file__).parent / "docs/sim14-south-carolina"
    os.makedirs(savedir, exist_ok=True)

    fname = savedir / "sc_wind_power.csv"
    raw_data = parse_raw_data(fname)

    cache_location = savedir / "cache"
    os.makedirs(cache_location, exist_ok=True)

    sample_size = 100
    test_frac = 0.5
    test_idx = int(sample_size * (1 - test_frac))
    num_samples = 1000
    payment = 1

    missing_probabilities = np.array([0, 0.5])

    imputation_methods = [
        ImputationMethod.no,
        ImputationMethod.mean,
        ImputationMethod.ols,
        # ImputationMethod.blr,
        ImputationMethod.mle,
        # ImputationMethod.gpr,
    ]

    @cache(save_dir=cache_location, use_cache=True)
    def _run_experiments():
        def _run_experiment():
            rho = 0.6
            # X = np.random.multivariate_normal(
            #     [0, 0, 0], [[1, 0, 0], [0, 1, rho], [0, rho, 1]], size=sample_size
            # )

            # X = np.column_stack((np.ones(len(X)).reshape(-1, 1), X))
            # y = (
            #     (X * coeffs).sum(axis=1)
            #     + np.random.normal(0, noise_variance**0.5, size=sample_size)
            # ).reshape(-1, 1)

            # print(X.shape, y.shape)

            n = sample_size  # for 2 random indices
            index = np.random.choice(len(raw_data), n, replace=False)

            y = raw_data["a1"].to_numpy().reshape(-1, 1)[index]
            X = raw_data[["a5", "a4", "a2"]].to_numpy()[index]
            X = np.column_stack((np.ones(len(X)).reshape(-1, 1), X))

            market_data = BatchData(
                dummy_feature=X[:, [0]],
                central_agent_features=X[:, [1]],
                support_agent_features=X[:, 2:],
                target_signal=y,
                test_frac=test_frac,
            )

            market = Market(market_data, GaussianProcessLinearRegression)
            losses, primary_market_payments, secondary_market_payments = market.run(
                imputation_methods=imputation_methods,
                missing_probabilities=missing_probabilities,
                payment=payment,
            )
            return losses, primary_market_payments, secondary_market_payments

        with tqdm_joblib(
            tqdm(desc="Simulations", total=num_samples, position=2, leave=False)
        ) as _:
            results = Parallel(n_jobs=-1)(
                delayed(_run_experiment)() for _ in range(num_samples)
            )

        return list(zip(*results))

    losses, primary_market_payments, secondary_market_payments = _run_experiments()

    label_map = {
        ImputationMethod.no: "No-missing",
        ImputationMethod.mean: "Mean imputation",
        ImputationMethod.ols: "Deterministic imputation",
        ImputationMethod.blr: "Probabilistic imputation",
        ImputationMethod.mle: "MLE imputation",
    }

    imputation_methods = [
        ImputationMethod.no,
        ImputationMethod.mean,
        ImputationMethod.ols,
        # ImputationMethod.blr,
        ImputationMethod.mle,
        # ImputationMethod.gpr,
    ]

    fig, ax = plt.subplots()

    for i, method in enumerate(imputation_methods):
        # loss = np.stack([l[method] for l in losses]) / np.stack(
        #     [l[ImputationMethod.no] for l in losses]
        # )
        print(i, method)
        loss = np.stack([l[method] for l in losses])

        num_runs = (np.arange(loss.shape[1]) + 1).reshape(-1, 1)
        # ax.plot(loss.cumsum(axis=1).mean(axis=0) / num_runs, label=label_map[method])
        # mean = loss.mean(axis=0).flatten()
        # upper = np.quantile()
        # upper = -loss.quantile(axis=0).flatten()
        # num_runs)
        mean = loss.mean(axis=0).flatten()
        upper = np.quantile(loss, q=0.95, axis=0).flatten()
        lower = np.quantile(loss, q=0.05, axis=0).flatten()
        med = np.quantile(loss, q=0.5, axis=0).flatten()
        # ax.fill_between(np.arange(len(mean)), upper, lower, color=f"C{i}", alpha=0.5)
        # ax.plot(med, color=f"C{i}")
        a = loss.mean(axis=1).flatten()
        # a = loss.flatten()
        x = np.sort(a)
        y = np.arange(len(x)) / float(len(x))
        ax.plot(x, y, alpha=1)

        # ax.hist(loss.flatten(), alpha=0.5, density=True)
    # ax.plot(loss.mean(axis=0), label=label_map[method])
    # ax.plot(loss.squeeze(), label=label_map[method], color=f"C{i}")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("E[Negative Log Likelihood]")
    # ax.legend()
    fig.tight_layout()
    fig.savefig(savedir / "losses", dpi=300)

    fig, axs = plt.subplots(2, 2, figsize=(10, 6), sharey=True)

    (axs1, axs2) = axs.flatten()[[0, 2]], axs.flatten()[[1, 3]]

    markers = ["*", "o", "s", "x"]

    metric = "payments"
    for i, method in enumerate(imputation_methods):
        primary_payments = np.stack([o[method] for o in primary_market_payments])
        secondary_payments = np.stack([o[method] for o in secondary_market_payments])
        num_runs = 1  # np.arange(payments.shape[1]) + 1

        j = 0
        axs.flatten()[0].plot(
            primary_payments.cumsum(axis=1).mean(axis=0)[:, j] / num_runs,
            label=label_map[method] if j == 0 else "",
            color=f"C{i}",
            # marker=markers[i],
        )
        axs.flatten()[1].plot(
            secondary_payments.cumsum(axis=1).mean(axis=0)[:, j] / num_runs,
            label="",
            color=f"C{i}",
            # marker=markers[i],
        )

        j = 1
        axs.flatten()[2].plot(
            primary_payments.cumsum(axis=1).mean(axis=0)[:, j] / num_runs,
            label=label_map[method] if j == 0 else "",
            color=f"C{i}",
            # marker=markers[i],
        )
        axs.flatten()[3].plot(
            secondary_payments.cumsum(axis=1).mean(axis=0)[:, j] / num_runs,
            label="",
            color=f"C{i}",
            # marker=markers[i],
        )

        axs.flatten()[0].set_title("Agent 1 (p(missing) = 0)")
        axs.flatten()[2].set_title("Agent 2 (p(missing) = 0.5)")
        axs.flatten()[1].set_title("Agent 1 (p(missing) = 0)")
        axs.flatten()[3].set_title("Agent 2 (p(missing) = 0.5)")

    for ax in axs.flatten():
        ax.grid()

    axs.flatten()[0].set_ylabel("E[C. Revenue]")
    axs.flatten()[2].set_ylabel("E[C. Revenue]")

    axs.flatten()[2].set_xlabel("Time step")
    axs.flatten()[3].set_xlabel("Time step")

    axs.flatten()[0].legend()

    fig.suptitle(
        "          Primary Market                                                       Secondary Market",
        y=0.97,
        fontsize=14,
    )

    fig.tight_layout()

    fig.savefig(savedir / "payments", dpi=300)
