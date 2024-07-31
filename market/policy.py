from typing import Dict, Tuple, Sequence, Set

import numpy as np

from market.task import Task
from common.utils import chain_combinations, safe_divide


class ShapleyAttributionPolicy:
    def __init__(
        self,
        active_agents: np.ndarray,
        baseline_agents: np.ndarray,
        regression_task: Task,
    ):
        self.active_agents = active_agents
        self.baseline_agents = baseline_agents
        self.num_active_agents = len(self.active_agents)
        self.num_baseline_agents = len(self.baseline_agents)
        self.num_agents = self.num_active_agents + self.num_baseline_agents
        self.regression_task = regression_task

    def _value(self, X: np.ndarray, y: np.ndarray, indices: Sequence, **kwargs) -> Dict:
        indices = list(indices)
        return self.regression_task.calculate_loss(X[:, indices], y, indices, **kwargs)

    def _marginal_contribution(
        self, X: np.ndarray, y: np.ndarray, excl: Set, incl: Set, **kwargs
    ) -> float:
        return self._value(X, y, list(excl), **kwargs) - self._value(
            X, y, list(incl), **kwargs
        )

    def _weighted_avg_contributions(
        self, X: np.ndarray, y: np.ndarray, X_covariance: np.ndarray
    ) -> np.ndarray:
        agent_combinations = chain_combinations(self.active_agents, 1, 1, replace=True)

        marginal_contributions = []
        for agent in self.active_agents:
            remaining_agents = self.active_agents.difference({agent})

            agent_marginal_contributions = []

            for combination in chain_combinations(
                remaining_agents, 0, len(remaining_agents)
            ):
                coalition_size = len(combination)

                # def coalition(with_agent: bool):
                #     coalition_agents = {
                #         agent_combinations.index(c) + len(self.baseline_agents)
                #         for c in agent_combinations
                #         if set(c).issubset(
                #             set(combination).union({agent} if with_agent else {})
                #         )
                #     }
                #     return list(self.baseline_agents | coalition_agents)

                coalition_agents = self.baseline_agents | {
                    agent_combinations.index(c) + len(self.baseline_agents)
                    for c in agent_combinations
                    if set(c).issubset(set(combination))
                }

                marginal_contribution = self._marginal_contribution(
                    X,
                    y,
                    coalition_agents,
                    coalition_agents.union({agent}),
                    X_covariance=X_covariance,
                )

                coalition_weight = (
                    np.math.factorial(coalition_size)
                    * np.math.factorial(self.num_active_agents - coalition_size - 1)
                    / np.math.factorial(self.num_active_agents)
                )

                agent_marginal_contributions.append(
                    coalition_weight * marginal_contribution
                )

            marginal_contributions.append(agent_marginal_contributions)

        return np.array(marginal_contributions).sum(axis=1)

    def run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        payment: float,
        X_covariance: np.ndarray = None,
    ) -> Tuple:
        contributions = self._weighted_avg_contributions(X, y, X_covariance)
        grand_coalition_contribution = self._marginal_contribution(
            X,
            y,
            self.baseline_agents,
            np.arange(self.num_agents),
            X_covariance=X_covariance,
        )
        allocations = safe_divide(contributions, grand_coalition_contribution)
        payments = contributions * len(X) * payment
        return contributions, allocations, payments
