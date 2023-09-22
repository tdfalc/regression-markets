from typing import List, Set, Dict, Tuple

import numpy as np
from scipy import stats

from market.task import Task
from common.utils import (
    chain_combinations,
    expected_kl_divergence_univariate_normal,
    safe_divide,
)


class SemivaluePolicy:
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
        raise NotImplementedError

    def _contribution(
        self, X: np.ndarray, y: np.ndarray, excl: Set, incl: Set
    ) -> float:
        return self._value(X, y, list(excl)) - self._value(X, y, list(incl))

    def _contribution_weight(self, *args) -> float:
        raise NotImplementedError

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
                    * self._contribution(X, y, coalition(False), coalition(True))
                )

            marginal_contributions.append(agent_marginal_contributions)
        return np.array(marginal_contributions).sum(axis=1)

    def _grand_coalition_contribution(self, X: np.ndarray, y: np.ndarray) -> float:
        return self._contribution(
            X, y, self.baseline_agents, np.arange(self.num_agents)
        )

    def _allocation(self, X: np.ndarray, y: np.ndarray, contributions: np.ndarray):
        grand_coalition_contribution = self._grand_coalition_contribution(X, y)
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
            results["loss_gc"] = self._value(X, y, np.arange(X.shape[1]))

            results["loss_buyer"] = self._value(
                X, y, np.arange(self.num_baseline_agents)
            )
        return results


class ShapleyPolicy(SemivaluePolicy):
    def __init__(
        self,
        active_agents: np.ndarray,
        baseline_agents: np.ndarray,
        polynomial_degree: int,
        regression_task: Task,
    ):
        super().__init__(
            active_agents=active_agents,
            baseline_agents=baseline_agents,
            polynomial_degree=polynomial_degree,
            regression_task=regression_task,
        )
        self._results = {}
        self.grand_coalition_size = len(self.active_agents)

    def _contribution_weight(self, coalition_size: int) -> float:
        return (
            np.math.factorial(coalition_size)
            * np.math.factorial(self.grand_coalition_size - coalition_size - 1)
            / np.math.factorial(self.grand_coalition_size)
        )


class NllShapleyPolicy(ShapleyPolicy):
    def __init__(
        self,
        active_agents: np.ndarray,
        baseline_agents: np.ndarray,
        polynomial_degree: int,
        regression_task: Task,
        observational: bool = True,
    ):
        super().__init__(
            active_agents=active_agents,
            baseline_agents=baseline_agents,
            polynomial_degree=polynomial_degree,
            regression_task=regression_task,
        )
        self.observational = observational

    def _value(self, X: np.ndarray, y: np.ndarray, indices: List) -> float:
        posterior = self._coalition_posterior(indices)
        noise_variance = self._coalition_noise_variance(indices)
        return self.regression_task.calculate_loss(
            X[:, np.sort(indices)], y, posterior, noise_variance
        )

    def _coalition_posterior(self, indices: List) -> np.ndarray:
        if self.observational:
            return self.regression_task.get_posterior(np.sort(indices))
        posterior = self.regression_task.get_posterior(np.arange(self.num_agents))
        mean = posterior.mean[np.sort(indices)]
        cov = posterior.cov[:, np.sort(indices)][np.sort(indices), :]

        return stats.multivariate_normal(mean, cov)

    def _coalition_noise_variance(self, indices: List) -> np.ndarray:
        if self.observational:
            return self.regression_task.get_noise_variance(np.sort(indices))
        # return self.regression_task.get_noise_variance(
        #     np.arange(self.num_agents)
        # )
        return self.regression_task.get_noise_variance(np.sort(indices))


class KldContModShapleyPolicy(NllShapleyPolicy):
    def __init__(
        self,
        active_agents: np.ndarray,
        baseline_agents: np.ndarray,
        polynomial_degree: int,
        regression_task: Task,
        observational: bool = True,
    ):
        super().__init__(
            active_agents=active_agents,
            baseline_agents=baseline_agents,
            polynomial_degree=polynomial_degree,
            regression_task=regression_task,
            observational=observational,
        )

    def _pred_mean_and_variance(self, X: np.ndarray, indices: List) -> Tuple:
        posterior = self._coalition_posterior(indices)
        noise_variance = self._coalition_noise_variance(indices)
        mean, variance = self.regression_task._predict(
            X[:, indices], posterior, noise_variance
        )
        return mean, variance

    def _value(self, X: np.ndarray, y: np.ndarray, indices: List) -> float:
        buyer_indices = np.arange(len(self.baseline_agents))
        buyer_mean, buyer_variance = self._pred_mean_and_variance(X, buyer_indices)
        coalition_mean, coalition_variance = self._pred_mean_and_variance(X, indices)
        return expected_kl_divergence_univariate_normal(
            buyer_mean, buyer_variance, coalition_mean, coalition_variance
        )

    def _contribution(
        self, X: np.ndarray, y: np.ndarray, excl: Set, incl: Set
    ) -> float:
        return self._value(X, y, list(incl)) - self._value(X, y, list(excl))


class KldContModShapleyPolicy2(NllShapleyPolicy):
    def __init__(
        self,
        active_agents: np.ndarray,
        baseline_agents: np.ndarray,
        polynomial_degree: int,
        regression_task: Task,
        observational: bool = True,
    ):
        super().__init__(
            active_agents=active_agents,
            baseline_agents=baseline_agents,
            polynomial_degree=polynomial_degree,
            regression_task=regression_task,
            observational=observational,
        )

    def _pred_mean_and_variance(self, X: np.ndarray, indices: List) -> Tuple:
        posterior = self._coalition_posterior(indices)
        noise_variance = self._coalition_noise_variance(indices)
        mean, variance = self.regression_task._predict(
            X[:, indices], posterior, noise_variance
        )
        return mean, variance

    def _value(self, X: np.ndarray, y: np.ndarray, indices: List) -> float:
        buyer_indices = np.arange(len(self.baseline_agents))
        gc_mean, gc_variance = self._pred_mean_and_variance(X, buyer_indices)
        coalition_mean, coalition_variance = self._pred_mean_and_variance(X, indices)
        return expected_kl_divergence_univariate_normal(
            coalition_mean, coalition_variance, gc_mean, gc_variance
        )

    def _contribution(
        self, X: np.ndarray, y: np.ndarray, excl: Set, incl: Set
    ) -> float:
        return self._value(X, y, list(incl)) - self._value(X, y, list(excl))


class KldCharModShapleyPolicy(KldContModShapleyPolicy):
    def __init__(
        self,
        active_agents: np.ndarray,
        baseline_agents: np.ndarray,
        polynomial_degree: int,
        regression_task: Task,
        observational: bool = True,
    ):
        super().__init__(
            active_agents=active_agents,
            baseline_agents=baseline_agents,
            polynomial_degree=polynomial_degree,
            regression_task=regression_task,
            observational=observational,
        )

    def _value(self, X: np.ndarray, y: np.ndarray, indices: List) -> float:
        pass

    def _contribution(
        self, X: np.ndarray, y: np.ndarray, excl: Set, incl: Set
    ) -> float:
        prior_mean, prior_variance = self._pred_mean_and_variance(X, list(excl))
        posterior_mean, posterior_variance = self._pred_mean_and_variance(X, list(incl))
        return expected_kl_divergence_univariate_normal(
            prior_mean, prior_variance, posterior_mean, posterior_variance
        )


class KldCharModShapleyPolicy2(KldContModShapleyPolicy):
    def __init__(
        self,
        active_agents: np.ndarray,
        baseline_agents: np.ndarray,
        polynomial_degree: int,
        regression_task: Task,
        observational: bool = True,
    ):
        super().__init__(
            active_agents=active_agents,
            baseline_agents=baseline_agents,
            polynomial_degree=polynomial_degree,
            regression_task=regression_task,
            observational=observational,
        )

    def _value(self, X: np.ndarray, y: np.ndarray, indices: List) -> float:
        pass

    def _contribution(
        self, X: np.ndarray, y: np.ndarray, excl: Set, incl: Set
    ) -> float:
        prior_mean, prior_variance = self._pred_mean_and_variance(X, list(excl))
        posterior_mean, posterior_variance = self._pred_mean_and_variance(X, list(incl))
        return expected_kl_divergence_univariate_normal(
            posterior_mean, posterior_variance, prior_mean, prior_variance
        )
