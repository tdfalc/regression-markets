from typing import List, Set, Dict, Tuple

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

    def _value(self, X: np.ndarray, y: np.ndarray, indices: List) -> Dict:
        return self.regression_task.calculate_loss(X, y, indices)

    def _marginal_contribution(
        self, X: np.ndarray, y: np.ndarray, excl: Set, incl: Set
    ) -> float:
        return self._value(X, y, list(excl)) - self._value(X, y, list(incl))

    def _contribution_weight(self, coalition_size: int) -> float:
        return (
            np.math.factorial(coalition_size)
            * np.math.factorial(self.num_active_agents - coalition_size - 1)
            / np.math.factorial(self.num_active_agents)
        )

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

                def coalition(with_agent: bool):
                    agents = set(combination).union({agent} if with_agent else {})
                    return self.baseline_agents | {
                        agent_combinations.index(c) + len(self.baseline_agents)
                        for c in agent_combinations
                        if set(c).issubset(agents)
                    }

                agent_marginal_contributions.append(
                    self._contribution_weight(len(combination))
                    * self._marginal_contribution(
                        X, y, coalition(False), coalition(True)
                    )
                )

            marginal_contributions.append(agent_marginal_contributions)
        return np.array(marginal_contributions).sum(axis=1)

    def _allocation(self, X: np.ndarray, y: np.ndarray, contributions: np.ndarray):
        grand_coalition_contribution = self._marginal_contribution(
            X, y, self.baseline_agents, np.arange(self.num_agents)
        )
        return safe_divide(contributions, grand_coalition_contribution)

    def run(self, X: np.ndarray, y: np.ndarray, payment: float) -> Tuple:
        results = {
            "allocations": np.full(self.num_active_agents, np.nan),
            "contributions": np.full(self.num_active_agents, np.nan),
            "payments": np.full(self.num_active_agents, np.nan),
        }
        if len(X) > 0:
            contributions = self._weighted_avg_contributions(X, y)
            results["allocations"] = self._allocation(X, y, contributions)
            results["contributions"] = contributions
            results["payments"] = contributions * len(X) * payment
        return results
