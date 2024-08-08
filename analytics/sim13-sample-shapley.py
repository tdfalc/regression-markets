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

import numpy as np
from joblib import Parallel, delayed
import itertools


def shapley_approx(
    agent: int,
    policy: SemivaluePolicy,
    X: np.ndarray,
    y: np.ndarray,
    sample_size: int,
):
    # Generate all permutations of the active agents
    permutations = list(itertools.permutations(policy.active_agents))

    def _contribution(permutation):
        # Determine the coalition up to the agent of interest
        agent_index = permutation.index(agent)
        agents_before = permutation[:agent_index]

        # Calculate value before adding the agent
        indices_before = list(policy.baseline_agents) + list(agents_before)
        value_before = policy._value(X, y, indices=indices_before)

        # Calculate value after adding the agent
        indices_after = indices_before + [agent]
        value_after = policy._value(X, y, indices=indices_after)

        return (value_before - value_after) / sample_size

    # Sample random permutations
    sampled_permutations = np.random.choice(
        len(permutations), size=sample_size, replace=True
    )

    # Compute contributions in parallel
    contributions = Parallel(n_jobs=-1, verbose=1)(
        delayed(_contribution)(permutations[i]) for i in sampled_permutations
    )

    return np.sum(contributions)


def shapley_approx(
    agent: int,
    permutations: list,
    policy: SemivaluePolicy,
    baseline_agents: set,
    X: np.ndarray,
    y: np.ndarray,
    sample_size: int,
):

    def _contribution(i):
        coalition = permutations[i]
        agents_before = []
        for j in coalition:
            if j == agent:
                break
            agents_before.append(j)
        # agents_before = [j for j in coalition if j != agent]
        indices = list(baseline_agents) + agents_before
        value_before = policy._value(X, y, indices=indices)
        indices += [agent]
        # indices = np.sort(indices)
        value_after = policy._value(X, y, indices=indices)
        return 1 * (value_before - value_after) / sample_size

    return np.sum(
        Parallel(n_jobs=-1, verbose=1)(
            delayed(_contribution)(i)
            for i in np.random.choice(np.arange(len(permutations)), size=sample_size)
        )
    )


# def shapley_approx(
#     agent: int,
#     permutations: list,
#     policy: SemivaluePolicy,
#     baseline_agents: set,
#     X: np.ndarray,
#     y: np.ndarray,
#     sample_size: int,
# ):

#     from sklearn.linear_model import LinearRegression

#     lr = LinearRegression(fit_intercept=False)
#     coeffs = lr.fit(X, y).coef_
#     return (coeffs.flatten()[agent] ** 2) * np.var(X[:, agent])


def shapley_approx(
    agent: int,
    policy: SemivaluePolicy,
    X: np.ndarray,
    y: np.ndarray,
    sample_size: int,
):

    # Generate permutations for each length from 1 to the length of the array
    permutations = [
        p
        for r in range(1, len(policy.active_agents) + 1)
        for p in itertools.permutations(policy.active_agents, r)
    ]

    def _contribution(permutation):
        # Determine the coalition up to the agent of interest
        agent_index = permutation.index(agent)
        agents_before = permutation[:agent_index]

        # Calculate value before adding the agent
        indices_before = list(policy.baseline_agents) + list(agents_before)
        value_before = policy._value(X, y, indices=indices_before)

        # Calculate value after adding the agent
        indices_after = indices_before + [agent]
        value_after = policy._value(X, y, indices=indices_after)

        return (value_before - value_after) / sample_size

    # Sample random permutations
    sampled_permutations = np.random.choice(
        len(permutations), size=sample_size, replace=True
    )

    # Compute contributions in parallel
    contributions = Parallel(n_jobs=-1, verbose=1)(
        delayed(_contribution)(permutations[i]) for i in sampled_permutations
    )

    return np.sum(contributions)


if __name__ == "__main__":

    logger = create_logger(__name__)
    logger.info("Running south carolina analysis")

    savedir = Path(__file__).parent / "docs/sim13-sample-shapley"
    os.makedirs(savedir, exist_ok=True)

    # np.random.seed(seed=42)

    train_size = 100000
    test_size = 100000
    coefficients = np.atleast_2d(
        [0.1, -0.5, 0.9, 0.9]
    ).T  # all additional features after 0, 1 are support agents

    noise_variance = 1.5
    regularization = 1e-32
    num_simulations = 1000
    train_payment = 0.03
    test_payment = 0.03

    # Well-specified model
    interpolant_function = lambda X: X @ coefficients
    additive_noise_function = lambda sample_size: np.random.normal(
        0,
        np.sqrt(noise_variance),
        size=(sample_size, 1),
    )
    heteroskedasticity_function = lambda X: 1

    sample_size = train_size + test_size
    test_frac = test_size / sample_size

    num_features = len(coefficients) - 1
    mean, covariance = np.zeros(num_features), np.eye(num_features)
    from sklearn.datasets import make_regression

    # X, y = make_regression(
    #     sample_size,
    #     n_features=4,
    #     n_informative=4,
    #     n_targets=1,
    #     effective_rank=1,
    #     noise=1,
    # )
    # y = y.reshape(-1, 1)
    covariance[-1, -2] = 0.99
    covariance[-2, -1] = 0.99
    from helpers import add_dummy

    X = np.random.multivariate_normal(mean, covariance, size=sample_size)
    X = add_dummy(X)
    noise = additive_noise_function(sample_size)
    y = interpolant_function(X)
    y += noise * heteroskedasticity_function(X)

    N = 1
    num_support = len(coefficients) - 2
    new_output = {
        "train": np.zeros((N, num_support)),
        "test": np.zeros((N, num_support)),
    }
    old_output = {
        "train": np.zeros((N, num_support)),
        "test": np.zeros((N, num_support)),
    }
    for SAMPLE in range(N):
        num_replications = np.random.randint(low=0, high=2, size=num_features - 1)
        num_replications = [0, 3]

        all_features_and_replicates = [X[:, 0:1], X[:, 1:2]]

        for i in range(2, len(coefficients)):
            replicates = []
            feat = X[:, i].reshape(-1, 1)
            all_features_and_replicates.append(feat)
            for r in range(num_replications[i - 2]):
                replicate = feat + np.random.normal(0, 0.12, size=(len(X), 1))
                all_features_and_replicates.append(replicate)

        X_new = np.hstack(all_features_and_replicates)

        data = BatchData(X[:, [0]], X[:, [1]], X[:, 2:], y, test_frac=test_frac)
        num_active_agents = len(data.active_agents)
        observational = True

        market = BatchMarket(
            data,
            MaximumLikelihoodLinearRegression(noise_variance=noise_variance),
            train_payment=train_payment,
            test_payment=test_payment,
            observational=observational,
        )
        policy = NllShapleyPolicy(
            active_agents=data.active_agents,
            baseline_agents=data.baseline_agents,
            polynomial_degree=data.degree,
            regression_task=market.regression_task,
            observational=market.observational,
        )
        # orig_output = market.run(NllShapleyPolicy)

        X, y = data.X_train, data.y_train
        sample_size = 100

        contributions = np.zeros(len(data.active_agents))
        for i, agent in enumerate(tqdm(data.active_agents)):
            contributions[i] = shapley_approx(
                agent,
                # permutations,
                policy,
                data.baseline_agents,
                data.active_agents,
                X,
                y,
                sample_size,
            )

        allocations = policy._allocation(X, y, contributions)
        allocations = contributions
        orig_output = {
            "train": {
                "allocations": allocations,
            },
            "test": {
                "allocations": allocations,
            },
        }

        data = BatchData(
            X_new[:, [0]], X_new[:, [1]], X_new[:, 2:], y, test_frac=test_frac
        )
        num_active_agents = len(data.active_agents)
        market = BatchMarket(
            data,
            MaximumLikelihoodLinearRegression(noise_variance=noise_variance),
            train_payment=train_payment,
            test_payment=test_payment,
            observational=observational,
        )
        policy = NllShapleyPolicy(
            active_agents=data.active_agents,
            baseline_agents=data.baseline_agents,
            polynomial_degree=data.degree,
            regression_task=market.regression_task,
            observational=market.observational,
        )
        # output = market.run(NllShapleyPolicy)

        X, y = data.X_train, data.y_train

        contributions = np.zeros(len(data.active_agents))
        for i, agent in enumerate(tqdm(data.active_agents)):
            contributions[i] = shapley_approx(
                agent,
                # permutations,
                policy,
                data.baseline_agents,
                data.active_agents,
                X,
                y,
                sample_size,
            )
        allocations = policy._allocation(X, y, contributions)
        allocations = contributions
        output = {
            "train": {
                "allocations": allocations,
            },
            "test": {
                "allocations": allocations,
            },
        }

        print(orig_output["train"])
        print(output["train"])

        k = 0
        for i in range(len(num_replications)):
            num_replicates = num_replications[i]

            old_output["train"][SAMPLE, i] = orig_output["train"]["allocations"][i]
            old_output["test"][SAMPLE, i] = orig_output["test"]["allocations"][i]

            new_output["train"][SAMPLE, i] = np.sum(
                output["train"]["allocations"][i + k : i + k + num_replicates + 1]
            )

            new_output["test"][SAMPLE, i] = np.sum(
                output["test"]["allocations"][i + k : i + k + num_replicates + 1]
            )

            k += num_replicates

    old_q50_train = np.quantile(old_output["train"], q=0.5, axis=0)
    old_mean_train = old_output["train"].mean(axis=0)
    new_mean_train = new_output["train"].mean(axis=0)
    new_q50_train = np.quantile(new_output["train"], q=0.5, axis=0)
    new_q95_train = np.quantile(new_output["train"], q=0.95, axis=0)
    new_q05_train = np.quantile(new_output["train"], q=0.05, axis=0)

    old_q50_test = np.quantile(old_output["test"], q=0.5, axis=0)
    old_mean_test = old_output["test"].mean(axis=0)
    new_mean_test = new_output["test"].mean(axis=0)
    new_q50_test = np.quantile(new_output["test"], q=0.5, axis=0)
    new_q95_test = np.quantile(new_output["test"], q=0.95, axis=0)
    new_q05_test = np.quantile(new_output["test"], q=0.05, axis=0)

    fig, ax = plt.subplots()

    # First we add total allocation bars
    positions = np.arange(len(old_q50_train))
    bar_width = 0.1
    color = "blue"
    hatch = "//"

    spacings = [-0.05, -0.025, 0.025, 0.05]
    for stage in ("train", "test"):

        for i, allocation in enumerate(
            (
                old_mean_train,
                old_mean_test,
                new_mean_train,
                new_mean_test,
            )
        ):
            xs = np.arange(len(old_mean_train)) + spacings[i]
            colors = ["darkorange", "blue", "black", "magenta"]
            if i < 2:
                ax.scatter(xs, allocation, c=colors[i])
            if i >= 2:
                ax.scatter(xs, allocation, c=colors[i])

    fig.savefig(savedir / "allocations.pdf")
