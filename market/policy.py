from typing import Dict, Tuple, Sequence

import numpy as np

from market.task import Task
from common.utils import chain_combinations, safe_divide


class ShapleyAttributionPolicy:
    def __init__(
        self,
        active_agents: np.ndarray,
        baseline_agents: np.ndarray,
        polynomial_degree: int,
        regression_task: Task,
    ):
        self.active_agents = active_agents
        self.baseline_agents = baseline_agents
        self.degree = polynomial_degree
        self.num_active_agents = len(self.active_agents)
        self.num_baseline_agents = len(self.baseline_agents)
        self.num_agents = self.num_active_agents + self.num_baseline_agents
        self.regression_task = regression_task

    def _value(self, X: np.ndarray, y: np.ndarray, indices: Sequence) -> Dict:
        return self.regression_task.calculate_loss(X, y, list(indices))

    def _weighted_avg_contributions(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        agent_combinations = chain_combinations(
            self.active_agents, 1, self.degree, replace=True
        )

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

                marginal_contribution = self._value(
                    X, y, coalition_agents
                ) - self._value(X, y, coalition_agents.union(agent))

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

    def run(self, X: np.ndarray, y: np.ndarray) -> Tuple:
        contributions = self._weighted_avg_contributions(X, y)
        grand_coalition_contribution = self._marginal_contribution(
            X, y, self.baseline_agents, np.arange(self.num_agents)
        )
        allocations = safe_divide(contributions, grand_coalition_contribution)
        # payments = contributions * len(X) * payment
        return contributions, allocations
