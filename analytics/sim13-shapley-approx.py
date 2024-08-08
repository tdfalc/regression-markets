import os
from pathlib import Path
from itertools import cycle
from typing import Tuple
import itertools

import numpy as np
from joblib import delayed, Parallel
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
from regression_markets.common.utils import chain_combinations, safe_divide

# It is unclear how scalable is the proposed solution with an increasing number of agents/observations/features.
# The proposed example is quite a toy contrary to all exaggerated statements on the relevance to real-life.
# Pls, provide theoretical analysis for analytical solutions and empirical results for approximate methods for larger problems.
# It would be nice to know when software/RAM bottlenecks kick in to clarify the practical value for realistic problems.
#  Consider close to the actual number of wind turbines of actual companies for a rather big region.

# (6) Is replication in the experiments with zero variance? Experiments on more datasets and having more than one
# agent with replication would make the paper stronger.


from regression_markets.market.data import BatchData
from regression_markets.market.task import BayesianLinearRegression
from regression_markets.market.mechanism import BatchMarket
from regression_markets.market.policy import NllShapleyPolicy
from regression_markets.common.log import create_logger
from regression_markets.common.utils import chain_combinations
from analytics.helpers import save_figure, conditional_value_at_risk, build_data
from regression_markets.common.utils import cache
from regression_markets.market.data import BatchData
from regression_markets.market.policy import (
    KldCfModShapleyPolicy,
    KldContributionModShapleyPolicy,
    NllShapleyPolicy,
    SemivaluePolicy,
)
from regression_markets.market.task import (
    BayesianLinearRegression,
    MaximumLikelihoodLinearRegression,
)
from helpers import add_dummy
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler

import numpy as np
from joblib import Parallel, delayed
import itertools
from helpers import tweak

import pandas as pd


def build_X_y(df, num_samples=None) -> BatchData:

    agents = ["a1", "a4", "a5", "a6", "a7", "a8", "a9"]
    agents = [
        "a1",
        "a4",
        "a5",
        "a6",
        "a7",
        "a8",
        "a9",
    ]
    # agents = ["a1", "a4", "a8"]
    central_agent = "a1"

    df = df.resample("1H").mean()
    df = df.loc[:, agents].copy()

    for agent in agents:
        df[f"{agent}_lag1"] = df[f"{agent}"].shift(1)
        if agent != central_agent:
            df = df.drop(f"{agent}", axis=1)

    # df /= df.max(axis=0)
    df = df.dropna()

    if num_samples is not None:
        idx = np.random.randint(0, len(df) - num_samples - 1)
        df = df.iloc[idx : idx + num_samples]

    target_signal = df.pop(f"{central_agent}").to_numpy().reshape(-1, 1)
    central_agent_cols = [f"{central_agent}_lag1"]
    central_agent_features = df.loc[:, central_agent_cols]
    support_agent_features = df.loc[
        :, [c for c in df.columns if c not in central_agent_cols]
    ]
    support_agent_features = support_agent_features.to_numpy()

    X = np.hstack(
        (
            np.ones((len(target_signal), 1)),
            central_agent_features.to_numpy(),
            support_agent_features,
        )
    )

    y = target_signal

    # return X, y

    return X, y


def parse_raw_data(fname: Path) -> pd.DataFrame:
    df = pd.read_csv(fname, header=None, index_col=0)
    df.index = pd.to_datetime(df.index, format="%d/%m/%Y %H.%M")
    df = df.tz_localize("utc")
    df.columns = [f"a{i}" for i in df.columns]
    return df


# def shapley_approx(
#     agent: int,
#     policy: SemivaluePolicy,
#     X: np.ndarray,
#     y: np.ndarray,
#     num_iterations: int,
# ):
#     """
#     Approximates the Shapley value of an agent using Monte Carlo sampling.

#     Args:
#         agent: The index of the agent whose Shapley value is being computed.
#         policy: A SemivaluePolicy object representing the policy.
#         X: The feature matrix.
#         y: The target variable.
#         num_iterations: The number of samples to use for the approximation.

#     Returns:
#         The approximate Shapley value of the agent.

#         DOESNT WORK but the below one does
#     """

#     active_agents = list(policy.active_agents)
#     baseline_agents = list(policy.baseline_agents)

#     def _contribution(permutation):
#         agents_before = baseline_agents + list(j for j in permutation if j < agent)
#         value_before = policy._value(X, y, indices=agents_before)
#         value_after = policy._value(X, y, indices=agents_before + [agent])
#         return value_before - value_after

#     permutations = []
#     for r in range(1, len(policy.active_agents) + 1):
#         perms = itertools.permutations(policy.active_agents, r)
#         permutations.extend(perms)

#     permutation_samples = np.random.choice(len(permutations), size=num_iterations)

#     return np.mean(
#         Parallel(n_jobs=-1, verbose=0)(
#             delayed(_contribution)(permutations[i]) for i in permutation_samples
#         )
#     )

from scipy import stats


def shapley_approx_perms(
    policy: SemivaluePolicy,
    X: np.ndarray,
    y: np.ndarray,
    sample_size: int,
    permutations,
):

    def _contribution(agent, coalition):
        agents_before = []
        for j in coalition:
            if j == agent:
                break
            agents_before.append(j)
        indices = list(policy.baseline_agents) + agents_before
        value_before = policy._value(X, y, indices=indices)
        indices += [agent]
        value_after = policy._value(X, y, indices=indices)
        return value_before - value_after

    contributions = np.zeros(len(policy.active_agents))
    for j, agent in enumerate(policy.active_agents):
        contributions[j] = np.mean(
            Parallel(n_jobs=-1, verbose=0)(
                delayed(_contribution)(agent, permutations[i])
                for i in np.random.choice(
                    np.arange(len(permutations)), size=sample_size
                )
            )
        )
    return policy._allocation(X, y, contributions)


# def shapley_approx(
#     policy: SemivaluePolicy,
#     X: np.ndarray,
#     y: np.ndarray,
#     sample_size: int,
# ):

#     # Generate permutations for each length from 1 to the length of the array
#     # permutations = []
#     # for r in range(1, len(policy.active_agents) + 1):
#     #     perms = itertools.permutations(policy.active_agents, r)
#     #     permutations.extend(perms)

#     permutations = list(
#         itertools.permutations(policy.active_agents, len(policy.active_agents))
#     )

#     def _contribution(agent, coalition):
#         agents_before = []
#         for j in coalition:
#             if j == agent:
#                 break
#             agents_before.append(j)
#         indices = list(policy.baseline_agents) + agents_before
#         value_before = policy._value(X, y, indices=indices)
#         indices += [agent]
#         value_after = policy._value(X, y, indices=indices)
#         return value_before - value_after

#     contributions = np.zeros(len(policy.active_agents))
#     for j, agent in enumerate(policy.active_agents):
#         contributions[j] = np.mean(
#             Parallel(n_jobs=-1, verbose=0)(
#                 delayed(_contribution)(agent, permutations[i])
#                 for i in np.random.choice(
#                     np.arange(len(permutations)), size=sample_size
#                 )
#             )
#         )
#     return policy._allocation(X, y, contributions)


def shapley_approx(
    policy: SemivaluePolicy,
    X: np.ndarray,
    y: np.ndarray,
    sample_size: int,
):

    # Generate permutations for each length from 1 to the length of the array
    # permutations = []
    # for r in range(1, len(policy.active_agents) + 1):
    #     perms = itertools.permutations(policy.active_agents, r)
    #     permutations.extend(perms)

    permutations = list(
        itertools.permutations(policy.active_agents, len(policy.active_agents))
    )

    # def _contribution(agent, coalition):
    #     agents_before = []
    #     for j in coalition:
    #         if j == agent:
    #             break
    #         agents_before.append(j)
    #     indices = list(policy.baseline_agents) + agents_before
    #     value_before = policy._value(X, y, indices=indices)
    #     indices += [agent]
    #     value_after = policy._value(X, y, indices=indices)
    #     return value_before - value_after

    def _contribution(coalition):
        indices_before = list(policy.baseline_agents)
        value_before = policy._value(X, y, indices=indices_before)
        contributions = np.zeros(len(policy.active_agents))
        for j, agent in enumerate(coalition):
            indices_after = indices_before + [agent]
            value_after = policy._value(X, y, indices=indices_after)
            contributions[agent - num_baseline_agents] += value_before - value_after
            value_before = value_after
            indices_before = indices_after
        return contributions

    contributions = np.zeros(len(policy.active_agents))
    sampled_permutations = np.random.choice(
        np.arange(len(permutations)), size=sample_size
    )
    num_baseline_agents = len(policy.baseline_agents)

    contributions = np.mean(
        Parallel(n_jobs=-1, verbose=1)(
            delayed(_contribution)(permutations[i]) for i in sampled_permutations
        ),
        axis=0,
    )

    return policy._allocation(X, y, contributions)


def shapley_det(
    policy: SemivaluePolicy,
    X: np.ndarray,
    y: np.ndarray,
):

    permutations = list(
        itertools.permutations(policy.active_agents, len(policy.active_agents))
    )

    def _contribution(coalition):
        indices_before = list(policy.baseline_agents)
        value_before = policy._value(X, y, indices=indices_before)
        contributions = np.zeros(len(policy.active_agents))
        for j, agent in enumerate(coalition):
            indices_after = indices_before + [agent]
            value_after = policy._value(X, y, indices=indices_after)
            contributions[agent - num_baseline_agents] += value_before - value_after
            value_before = value_after
            indices_before = indices_after
        return contributions

    contributions = np.zeros(len(policy.active_agents))

    num_baseline_agents = len(policy.baseline_agents)

    contributions = np.mean(
        Parallel(n_jobs=-1, verbose=1)(
            delayed(_contribution)(permutation) for permutation in permutations
        ),
        axis=0,
    )

    return policy._allocation(X, y, contributions)


def shapley_approx2(
    active_agents,
    baseline_agents,
    regression_task,
    observational,
    X: np.ndarray,
    y: np.ndarray,
    sample_size: int,
):

    num_active_agents = len(active_agents)
    num_baseline_agents = len(baseline_agents)
    num_agents = num_active_agents + num_baseline_agents

    permutations = list(itertools.permutations(active_agents, len(active_agents)))

    regression_task.update_posterior(X, y, np.arange(num_agents))

    def _coalition_posterior(indices) -> np.ndarray:
        regression_task.update_posterior(X, y, np.sort(indices))

        if observational:
            return regression_task.get_posterior(np.sort(indices))
        posterior = regression_task.get_posterior(np.arange(num_agents))
        mean = posterior.mean[np.sort(indices)]
        cov = posterior.cov[:, np.sort(indices)][np.sort(indices), :]

        return stats.multivariate_normal(mean, cov)

    def _coalition_noise_variance(indices) -> np.ndarray:
        if observational:
            return regression_task.get_noise_variance(np.sort(indices))
        return regression_task.get_noise_variance(np.sort(indices))

    def _value(X, y, indices):
        posterior = _coalition_posterior(indices)
        noise_variance = _coalition_noise_variance(indices)
        return regression_task.calculate_loss(
            X[:, np.sort(indices)], y, posterior, noise_variance
        )

    def _contribution(coalition):
        indices_before = list(baseline_agents)
        value_before = _value(X, y, indices=indices_before)
        contributions = np.zeros(len(active_agents))
        for j, agent in enumerate(coalition):
            indices_after = indices_before + [agent]
            value_after = _value(X, y, indices=indices_after)
            contributions[agent - num_baseline_agents] += value_before - value_after
            value_before = value_after
            indices_before = indices_after
        return contributions

    contributions = np.zeros(len(active_agents))
    sampled_permutations = np.random.choice(
        np.arange(len(permutations)), size=sample_size
    )
    num_baseline_agents = len(baseline_agents)

    contributions = np.mean(
        Parallel(n_jobs=-1, verbose=1)(
            delayed(_contribution)(permutations[i]) for i in sampled_permutations
        ),
        axis=0,
    )
    grand_coalition_contribution = _value(X, y, list(baseline_agents)) - _value(
        X, y, list(np.arange(num_agents))
    )

    return safe_divide(contributions, grand_coalition_contribution)


def noise_variance_mle(data: BatchData) -> float:
    task = MaximumLikelihoodLinearRegression()
    indices = np.arange(data.X.shape[1])
    task.update_posterior(data.X, data.y, indices)
    return task.get_noise_variance(indices)


if __name__ == "__main__":
    logger = create_logger(__name__)
    logger.info("Running south carolina analysis")

    savedir = Path(__file__).parent / "docs/sim13-shapley-approx"
    os.makedirs(savedir, exist_ok=True)

    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.size"] = 12
    plt.rcParams["mathtext.fontset"] = "cm"  # Use CM for math font.
    plt.rcParams["figure.autolayout"] = True  # Use tight layouts.

    test_frac = 0.1

    # np.random.seed(123)

    #
    regularization = 1e-32

    # observational = False
    # market_data = BatchData(X[:, [0]], X[:, [1]], X[:, 2:], y, test_frac=test_frac)
    # market = BatchMarket(
    #     market_data,
    #     MaximumLikelihoodLinearRegression(noise_variance=noise_variance),
    #     observational=observational,
    # )
    # policy = NllShapleyPolicy(
    #     active_agents=market_data.active_agents,
    #     baseline_agents=market_data.baseline_agents,
    #     polynomial_degree=market_data.degree,
    #     regression_task=market.regression_task,
    #     observational=market.observational,
    # )

    # # Experiment 1 - Convergence

    # train_size = 1000
    # test_size = 1000
    # coefficients = np.atleast_2d([0.1, -0.5, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]).T
    # # coefficients = np.atleast_2d([0.1, -0.5, 0.9, 0.9]).T

    # noise_variance = 1
    # sample_size = train_size + test_size
    # test_frac = test_size / sample_size

    # num_features = len(coefficients) - 1
    # mean, covariance = np.zeros(num_features), np.eye(num_features)
    # # covariance[-1, -2] = 0.99
    # # covariance[-2, -1] = 0.99
    # X = np.random.multivariate_normal(mean, covariance, size=sample_size)
    # X = add_dummy(X)
    # y = X @ coefficients + np.random.normal(
    #     0,
    #     np.sqrt(noise_variance),
    #     size=(sample_size, 1),
    # )

    # observational = False
    # market_data = BatchData(X[:, [0]], X[:, [1]], X[:, 2:], y, test_frac=test_frac)
    # market = BatchMarket(
    #     market_data,
    #     MaximumLikelihoodLinearRegression(noise_variance=noise_variance),
    #     observational=observational,
    # )
    # policy = NllShapleyPolicy(
    #     active_agents=market_data.active_agents,
    #     baseline_agents=market_data.baseline_agents,
    #     polynomial_degree=market_data.degree,
    #     regression_task=market.regression_task,
    #     observational=market.observational,
    # )

    # all_num_iterations = np.geomspace(1, 10000, 5).astype(int)
    # print(all_num_iterations)
    # num_active_agents = len(policy.active_agents)

    # eps = 0.001
    # delta = 0.99
    # print(num_active_agents * np.log(2 / delta) / (2 * (eps**2)))

    # results = {}
    # # results["deterministic"] = market.run(NllShapleyPolicy)["train"]["allocations"]
    # results["approximation"] = np.zeros((len(all_num_iterations), num_active_agents))

    # for i, num_iterations in enumerate(tqdm(all_num_iterations)):
    #     results["approximation"][i] = shapley_approx(
    #         policy, market_data.X_train, market_data.y_train, num_iterations
    #     )

    # fig, ax = plt.subplots()
    # for j, agent in enumerate(market_data.active_agents):
    #     ax.plot(all_num_iterations, results["approximation"][:, j])
    #     # ax.axhline(y=results["deterministic"][j], ls="--", c="k")
    # ax.set_xlabel("Sample Size")
    # ax.set_xscale("log")
    # fig.tight_layout()
    # fig.savefig(savedir / "convergence.pdf")

    # # Experiment 2 - Random replicates
    # num_iters = 1000
    # num_runs = 200
    # num_active_agents = len(market_data.active_agents)
    # contributions_before = np.zeros((num_runs, num_active_agents))
    # contributions_after = np.zeros((num_runs, num_active_agents))

    # contributions_before = {
    #     "observational": np.zeros((num_runs, num_active_agents)),
    #     "interventional": np.zeros((num_runs, num_active_agents)),
    # }
    # contributions_after = {
    #     "observational": np.zeros((num_runs, num_active_agents)),
    #     "interventional": np.zeros((num_runs, num_active_agents)),
    # }

    # baseline_agents = list(policy.baseline_agents)
    # active_agents = list(policy.active_agents)

    # for run in tqdm(range(num_runs)):

    #     for method in ("observational", "interventional"):
    #         num_replications = [0, 3]
    #         num_replications = np.random.randint(low=0, high=3, size=num_active_agents)

    #         all_features_and_replicates = [X[:, baseline_agents]]
    #         for j, agent in enumerate(active_agents):
    #             replicates = []
    #             feat = X[:, agent : agent + 1]
    #             all_features_and_replicates.append(feat)
    #             for r in range(num_replications[j]):
    #                 replicate = feat + np.random.normal(0, 0.12, size=(len(X), 1))
    #                 all_features_and_replicates.append(replicate)

    #         X_new = np.hstack(all_features_and_replicates)

    #         def calculate_contributions(X, y, test_frac, observational):

    #             market_data = BatchData(
    #                 X[:, [0]], X[:, [1]], X[:, 2:], y, test_frac=test_frac
    #             )
    #             num_active_agents = len(market_data.active_agents)
    #             task = MaximumLikelihoodLinearRegression(noise_variance=noise_variance)
    #             market = BatchMarket(market_data, task, observational=observational)
    #             policy = NllShapleyPolicy(
    #                 active_agents=market_data.active_agents,
    #                 baseline_agents=market_data.baseline_agents,
    #                 polynomial_degree=market_data.degree,
    #                 regression_task=market.regression_task,
    #                 observational=market.observational,
    #             )
    #             return shapley_approx(policy, X, y, num_iters)
    #             # return market.run(NllShapleyPolicy)["train"]["allocations"]

    #         observational = True if method == "observational" else False
    #         contributions_before[method][run] = calculate_contributions(
    #             X, y, test_frac, observational
    #         )
    #         # print(run, method, num_replications, contributions_before[method][run])
    #         # print(X[:5])
    #         # print(X_new[:5])
    #         ouput_after = calculate_contributions(X_new, y, test_frac, observational)

    #         ## this part is wrong
    #         k = 0
    #         for i in range(len(num_replications)):

    #             num_replicates = num_replications[i]
    #             contributions_after[method][run, i] = np.sum(
    #                 ouput_after[i + k : i + k + num_replicates + 1]
    #             )

    #             # print("$$", i, k, num_replicates, ouput_after)

    #             k += num_replicates

    #         # print(run, method, num_replications, contributions_after[method][run])

    # fig, ax = plt.subplots()

    # # First we add total allocation bars
    # bar_width = 0.1
    # color = "blue"
    # hatch = "//"

    # spacings = np.array([-0.05, -0.025, 0.025, 0.05])

    # # print(contributions_before)
    # # print(contributions_after)

    # xs = np.arange(num_active_agents) + spacings[i]
    # ax.errorbar(
    #     xs - 0.05,
    #     np.quantile(contributions_before["observational"], q=0.5, axis=0),
    #     yerr=(
    #         np.quantile(contributions_before["observational"], q=0.5, axis=0)
    #         - np.quantile(contributions_before["observational"], q=0.01, axis=0),
    #         np.quantile(contributions_before["observational"], q=0.99, axis=0)
    #         - np.quantile(contributions_before["observational"], q=0.5, axis=0),
    #     ),
    #     fmt="o",
    #     color="darkorange",
    # )
    # ax.errorbar(
    #     xs - 0.025,
    #     np.quantile(contributions_after["observational"], q=0.5, axis=0),
    #     yerr=(
    #         np.quantile(contributions_after["observational"], q=0.5, axis=0)
    #         - np.quantile(contributions_after["observational"], q=0.01, axis=0),
    #         np.quantile(contributions_after["observational"], q=0.99, axis=0)
    #         - np.quantile(contributions_after["observational"], q=0.5, axis=0),
    #     ),
    #     fmt="o",
    #     color="blue",
    # )
    # ax.errorbar(
    #     xs + 0.025,
    #     np.quantile(contributions_before["interventional"], q=0.5, axis=0),
    #     yerr=(
    #         np.quantile(contributions_before["interventional"], q=0.5, axis=0)
    #         - np.quantile(contributions_before["interventional"], q=0.01, axis=0),
    #         np.quantile(contributions_before["interventional"], q=0.99, axis=0)
    #         - np.quantile(contributions_before["interventional"], q=0.5, axis=0),
    #     ),
    #     fmt="o",
    #     color="magenta",
    # )
    # ax.errorbar(
    #     xs + 0.05,
    #     np.quantile(contributions_after["interventional"], q=0.5, axis=0),
    #     yerr=(
    #         np.quantile(contributions_after["interventional"], q=0.5, axis=0)
    #         - np.quantile(contributions_after["interventional"], q=0.01, axis=0),
    #         np.quantile(contributions_after["interventional"], q=0.99, axis=0)
    #         - np.quantile(contributions_after["interventional"], q=0.5, axis=0),
    #     ),
    #     fmt="o",
    #     color="black",
    # )

    # fig.savefig(savedir / "allocations_real.pdf")

    # # Experiment 3
    # cache_location = savedir / "cache"
    # os.makedirs(cache_location, exist_ok=True)

    # @cache(cache_location)
    # def _run_experiments():
    #     fname = savedir / "sc_wind_power.csv"
    #     raw_data = parse_raw_data(fname)
    #     X, y = build_X_y(raw_data)

    #     num_iters = 100
    #     num_runs = 10
    #     market_data = BatchData(X[:, [0]], X[:, [1]], X[:, 2:], y, test_frac=test_frac)
    #     noise_variance = noise_variance_mle(market_data)
    #     task = MaximumLikelihoodLinearRegression(noise_variance=noise_variance)
    #     task = BayesianLinearRegression(
    #         noise_variance=noise_variance, regularization=regularization
    #     )
    #     market = BatchMarket(market_data, task, observational=True)
    #     policy = NllShapleyPolicy(
    #         active_agents=market_data.active_agents,
    #         baseline_agents=market_data.baseline_agents,
    #         polynomial_degree=market_data.degree,
    #         regression_task=market.regression_task,
    #         observational=market.observational,
    #     )
    #     num_active_agents = len(policy.active_agents)
    #     contributions_before = np.zeros((num_runs, num_active_agents))
    #     contributions_after = np.zeros((num_runs, num_active_agents))

    #     contributions_before = {
    #         "observational": np.zeros((num_runs, num_active_agents)),
    #         "interventional": np.zeros((num_runs, num_active_agents)),
    #     }
    #     contributions_after = {
    #         "observational": np.zeros((num_runs, num_active_agents)),
    #         "interventional": np.zeros((num_runs, num_active_agents)),
    #     }

    #     baseline_agents = list(policy.baseline_agents)
    #     active_agents = list(policy.active_agents)

    #     for run in tqdm(range(num_runs)):
    #         num_replications = [0, 3]
    #         num_replications = np.random.randint(low=0, high=5, size=num_active_agents)
    #         # num_replications[3:] = 0
    #         print(num_replications)
    #         # num_replications = np.zeros(num_active_agents).astype(int)

    #         for method in ("observational", "interventional"):

    #             all_features_and_replicates = [X[:, baseline_agents]]
    #             for j, agent in enumerate(active_agents):
    #                 feat = X[:, agent : agent + 1]
    #                 all_features_and_replicates.append(feat)
    #                 for _ in range(num_replications[j]):
    #                     replicate = feat + np.random.normal(0, 0.12, size=(len(X), 1))
    #                     all_features_and_replicates.append(replicate)

    #             X_new = np.hstack(all_features_and_replicates)

    #             def calculate_contributions(X, y, test_frac, observational):

    #                 market_data = BatchData(
    #                     X[:, [0]], X[:, [1]], X[:, 2:], y, test_frac=test_frac
    #                 )

    #                 noise_variance = noise_variance_mle(market_data)
    #                 task = BayesianLinearRegression(
    #                     noise_variance=noise_variance, regularization=regularization
    #                 )
    #                 market = BatchMarket(market_data, task, observational=observational)
    #                 policy = NllShapleyPolicy(
    #                     active_agents=market_data.active_agents,
    #                     baseline_agents=market_data.baseline_agents,
    #                     polynomial_degree=market_data.degree,
    #                     regression_task=market.regression_task,
    #                     observational=market.observational,
    #                 )
    #                 return shapley_approx(policy, X, y, num_iters)
    #                 # return market.run(NllShapleyPolicy)["train"]["allocations"] * 100

    #             observational = True if method == "observational" else False
    #             contributions_before[method][run] = calculate_contributions(
    #                 X, y, test_frac, observational
    #             )

    #             ouput_after = calculate_contributions(
    #                 X_new, y, test_frac, observational
    #             )
    #             k = 0
    #             for i in range(len(num_replications)):

    #                 num_replicates = num_replications[i]
    #                 contributions_after[method][run, i] = np.sum(
    #                     ouput_after[i + k : i + k + num_replicates + 1]
    #                 )
    #                 k += num_replicates

    #     return contributions_before, contributions_after

    # contributions_before, contributions_after = _run_experiments()

    # fig, ax = plt.subplots(figsize=(4, 3), dpi=300)

    # # First we add total allocation bars
    # bar_width = 0.1
    # color = "blue"
    # hatch = "//"

    # spacings = np.array(
    #     [
    #         # -0.3,
    #         -0.15,
    #         0.15,
    #         # 0.3,
    #     ]
    # )

    # num_active_agents = 6

    # xs = np.arange(num_active_agents)  # + spacings[i]
    # lower, upper = 1e-6, 1 - 1e-6

    # colors = ["darkorange", "blue", "magenta", "green"]

    # for i, method in enumerate(
    #     (
    #         # contributions_before["observational"],
    #         # contributions_after["observational"],
    #         # contributions_before["interventional"],
    #         # contributions_after["interventional"],
    #         "observational",
    #         "interventional",
    #     )
    # ):
    #     allocations = (
    #         contributions_before[method] * 100 - contributions_after[method] * 100
    #     )
    #     # allocations = contributions_after[method] * 100
    #     ax.errorbar(
    #         xs + spacings[i],
    #         np.quantile(allocations, q=0.5, axis=0),
    #         yerr=(
    #             np.quantile(allocations, q=0.5, axis=0)
    #             - np.quantile(allocations, q=lower, axis=0),
    #             np.quantile(allocations, q=upper, axis=0)
    #             - np.quantile(allocations, q=0.5, axis=0),
    #         ),
    #         fmt="o",
    #         color=colors[i],
    #         # marker=marker,
    #         capsize=3,
    #         # color=color,
    #         # label=market_design,
    #         # markerfacecolor=color,
    #         markersize=6,
    #         lw=1,
    #         markeredgewidth=0.8,
    #         markeredgecolor="k",
    #         label=method.capitalize(),
    #     )

    # ax.set_xticks(np.arange(6))

    # ax.set_xticklabels(["$a_4$", "$a_5$", "$a_6$", "$a_7$", "$a_8$", "$a_9$"])
    # tweak(ax=ax, legend_loc="lower right")
    # ax.set_xlabel("Support Agent")
    # ax.set_ylabel("Revenue Allocation (\%)")

    # fig.savefig(savedir / "allocations_real.pdf")

    # Experiment 4 - uniform increase in num_repliactes
    cache_location = savedir / "cache"
    os.makedirs(cache_location, exist_ok=True)

    @cache(cache_location)
    def _run_experiments():
        fname = savedir / "sc_wind_power.csv"
        raw_data = parse_raw_data(fname)
        X, y = build_X_y(raw_data)
        X = X[:, [0, 1, 2, 3, 4, 5]]

        num_iters = 100

        #
        max_replicates = 3
        market_data = BatchData(X[:, [0]], X[:, [1]], X[:, 2:], y, test_frac=test_frac)
        noise_variance = noise_variance_mle(market_data)
        task = MaximumLikelihoodLinearRegression(noise_variance=noise_variance)
        task = BayesianLinearRegression(
            noise_variance=noise_variance, regularization=regularization
        )
        market = BatchMarket(market_data, task, observational=True)
        policy = NllShapleyPolicy(
            active_agents=market_data.active_agents,
            baseline_agents=market_data.baseline_agents,
            polynomial_degree=market_data.degree,
            regression_task=market.regression_task,
            observational=market.observational,
        )
        num_active_agents = len(policy.active_agents)
        contributions_before = np.zeros((max_replicates, num_active_agents))
        contributions_after = np.zeros((max_replicates, num_active_agents))

        contributions_before = {
            "observational": np.zeros((max_replicates, num_active_agents)),
            "interventional": np.zeros((max_replicates, num_active_agents)),
        }
        contributions_after = {
            "observational": np.zeros((max_replicates, num_active_agents)),
            "interventional": np.zeros((max_replicates, num_active_agents)),
        }

        baseline_agents = list(policy.baseline_agents)
        active_agents = list(policy.active_agents)

        for n in tqdm(range(max_replicates)):

            num_replications = (n + np.zeros(num_active_agents)).astype(int)
            # num_replications[-1] = 0
            print(num_replications)

            for method in ("observational", "interventional"):

                all_features_and_replicates = [X[:, baseline_agents]]
                for j, agent in enumerate(active_agents):
                    feat = X[:, agent : agent + 1]
                    all_features_and_replicates.append(feat)
                    for _ in range(num_replications[j]):
                        replicate = feat + np.random.normal(0, 0.12, size=(len(X), 1))
                        all_features_and_replicates.append(replicate)

                X_new = np.hstack(all_features_and_replicates)

                def calculate_contributions(X, y, test_frac, observational):

                    market_data = BatchData(
                        X[:, [0]], X[:, [1]], X[:, 2:], y, test_frac=test_frac
                    )

                    task = MaximumLikelihoodLinearRegression()

                    noise_variance = noise_variance_mle(market_data)
                    task = BayesianLinearRegression(
                        noise_variance=noise_variance, regularization=regularization
                    )
                    # market = BatchMarket(market_data, task, observational=observational)
                    # policy = NllShapleyPolicy(
                    #     active_agents=market_data.active_agents,
                    #     baseline_agents=market_data.baseline_agents,
                    #     polynomial_degree=market_data.degree,
                    #     regression_task=market.regression_task,
                    #     observational=market.observational,
                    # )
                    # return shapley_approx(policy, X, y, num_iters)
                    # return market.run(NllShapleyPolicy)["train"]["allocations"]

                    return shapley_approx2(
                        market_data.active_agents,
                        market_data.baseline_agents,
                        task,
                        observational,
                        market_data.X_train,
                        market_data.y_train,
                        num_iters,
                    )

                observational = True if method == "observational" else False
                contributions_before[method][n] = calculate_contributions(
                    X, y, test_frac, observational
                )

                ouput_after = calculate_contributions(
                    X_new, y, test_frac, observational
                )
                k = 0
                for i in range(len(num_replications)):

                    num_replicates = num_replications[i]
                    contributions_after[method][n, i] = np.sum(
                        ouput_after[i + k : i + k + num_replicates + 1]
                    )
                    k += num_replicates

        return contributions_before, contributions_after

    contributions_before, contributions_after = _run_experiments()

    fig, ax = plt.subplots(figsize=(4, 3), dpi=300)

    # First we add total allocation bars
    bar_width = 0.1
    color = "blue"
    hatch = "//"

    spacings = np.array(
        [
            # -0.3,
            -0.15,
            0.15,
            # 0.3,
        ]
    )

    num_active_agents = 6

    xs = np.arange(num_active_agents)  # + spacings[i]
    lower, upper = 1e-6, 1 - 1e-6

    colors = ["darkorange", "blue", "magenta", "green"]

    for i, method in enumerate(
        (
            "observational",
            "interventional",
        )
    ):
        allocations = (
            contributions_before[method] * 100 - contributions_after[method] * 100
        )
        allocations = contributions_after[method] * 100
        for j in range(2):
            ax.plot(
                allocations[:, j], color=colors[i], label=method if j == 0 else None
            )

    tweak(ax=ax, legend_loc="lower right")
    ax.set_ylabel("Revenue Allocation (\%)")
    fig.savefig(savedir / "allocations_real.pdf")

    # Experiment 5 - convergenece with wind data

    # fname = savedir / "sc_wind_power.csv"
    # raw_data = parse_raw_data(fname)
    # X, y = build_X_y(raw_data)
    # observational = False
    # test_frac = 0.1

    # market_data = BatchData(X[:, [0]], X[:, [1]], X[:, 2:], y, test_frac=test_frac)
    # noise_variance = noise_variance_mle(market_data)
    # task = BayesianLinearRegression(
    #     noise_variance=noise_variance, regularization=regularization
    # )
    # market = BatchMarket(market_data, task, observational=observational)
    # policy = NllShapleyPolicy(
    #     active_agents=market_data.active_agents,
    #     baseline_agents=market_data.baseline_agents,
    #     polynomial_degree=market_data.degree,
    #     regression_task=market.regression_task,
    #     observational=market.observational,
    # )
    # num_active_agents = len(policy.active_agents)

    # all_num_iterations = np.geomspace(1, 10000, 4).astype(int)
    # num_active_agents = len(policy.active_agents)

    # eps = 0.001
    # delta = 0.99
    # print(num_active_agents * np.log(2 / delta) / (2 * (eps**2)))

    # results = {}
    # results["deterministic"] = market.run(NllShapleyPolicy)["train"]["allocations"]
    # results["approximation"] = np.zeros((len(all_num_iterations), num_active_agents))

    # for i, num_iterations in enumerate(tqdm(all_num_iterations)):
    #     results["approximation"][i] = shapley_approx(
    #         policy, market_data.X_train, market_data.y_train, num_iterations
    #     )

    # fig, ax = plt.subplots()
    # for j, agent in enumerate(market_data.active_agents):
    #     ax.plot(all_num_iterations, results["approximation"][:, j])
    #     ax.axhline(y=results["deterministic"][j], ls="--", c="k")
    # ax.set_xlabel("Sample Size")
    # ax.set_xscale("log")
    # fig.tight_layout()
    # fig.savefig(savedir / "convergence.pdf")

    # # Experiment 6 - computation time

    # import time

    # fname = savedir / "sc_wind_power.csv"
    # raw_data = parse_raw_data(fname)
    # X, y = build_X_y(raw_data, num_samples=2000)
    # observational = False
    # test_frac = 0.5

    # delta = 0.01
    # eps = 0.05
    # factor = np.log(2 / delta) / (2 * (eps**2))

    # cache_location = savedir / "cache"
    # os.makedirs(cache_location, exist_ok=True)

    # num_features = np.arange(1, 8)
    # print(num_features)

    # @cache(cache_location)
    # def _run_experiments():
    #     results = {}
    #     results["deterministic"] = np.zeros(len(num_features))
    #     results["approximation"] = np.zeros(len(num_features))

    #     for i, M in enumerate(tqdm(num_features)):

    #         X_new = X[:, [0, 1]]
    #         X_features = np.random.multivariate_normal(
    #             np.zeros(M), np.eye(M), size=len(X)
    #         )
    #         X_new = np.hstack((X_new, X_features))

    #         market_data = BatchData(
    #             X_new[:, [0]], X_new[:, [1]], X_new[:, 2:], y, test_frac=test_frac
    #         )
    #         noise_variance = noise_variance_mle(market_data)
    #         task = BayesianLinearRegression(
    #             noise_variance=noise_variance, regularization=regularization
    #         )
    #         market = BatchMarket(market_data, task, observational=observational)
    #         policy = NllShapleyPolicy(
    #             active_agents=market_data.active_agents,
    #             baseline_agents=market_data.baseline_agents,
    #             polynomial_degree=market_data.degree,
    #             regression_task=market.regression_task,
    #             observational=market.observational,
    #         )

    #         delta = 0.01
    #         eps = 0.1
    #         factor = int(np.log10(2 / delta) / (2 * (eps**2)))

    #         sample_size = 10  # int(M * factor)

    #         print(i, M, sample_size)

    #         tic = time.perf_counter()
    #         shapley_approx(
    #             policy,
    #             market_data.X_train,
    #             market_data.y_train,
    #             sample_size=sample_size,
    #         )
    #         toc = time.perf_counter()
    #         results["approximation"][i] = toc - tic

    #         tic = time.perf_counter()
    #         # market.run(NllShapleyPolicy)
    #         shapley_det(
    #             policy,
    #             market_data.X_train,
    #             market_data.y_train,
    #         )
    #         toc = time.perf_counter()
    #         results["deterministic"][i] = toc - tic
    #     return results

    # results = _run_experiments()

    # fig, ax = plt.subplots()
    # ax.plot(num_features, results["approximation"], label="Approx")
    # ax.plot(num_features, results["deterministic"], label="Det")

    # # ax.set_xlabel("Sample Size")
    # ax.set_yscale("log")
    # tweak(ax=ax)
    # fig.tight_layout()
    # fig.savefig(savedir / "convergence.pdf")
