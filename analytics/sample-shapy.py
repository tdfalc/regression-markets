import os
from pathlib import Path
from itertools import cycle
from typing import Tuple

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


def shapley_approx(
    agent: int,
    policy: SemivaluePolicy,
    baseline_agents: set,
    active_agents,
    X: np.ndarray,
    y: np.ndarray,
    sample_size: int,
):

    # Generate permutations for each length from 1 to the length of the array
    permutations = []
    for r in range(1, len(active_agents) + 1):
        perms = itertools.permutations(active_agents, r)
        permutations.extend(perms)

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


if __name__ == "__main__":

    np.random.seed(seed=42)

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
    orig_output = market.run(NllShapleyPolicy)
    print(orig_output["train"])

    X, y = data.X_train, data.y_train
    sample_size = 1000
    permutations = chain_combinations(
        data.active_agents, 1, num_active_agents, replace=True
    )

    import itertools

    contributions = np.zeros(len(data.active_agents))
    for i, agent in enumerate(tqdm(data.active_agents)):
        contributions[i] = shapley_approx(
            agent, policy, data.baseline_agents, data.active_agents, X, y, sample_size
        )
    print(contributions)
    allocations = policy._allocation(X, y, contributions)
    print(allocations)
