from typing import Tuple, Sequence, List, Dict
import itertools

import numpy as np
from tqdm import tqdm

from allocation.model import Model
from allocation.state import StateTracker


class SemivaluePolicy:
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        forecast_model: Model,
        dummy_agents: Tuple[int],
        central_agents: Tuple[int],
        support_agents: Tuple[int],
        polynomial_order: int = 1,
    ):
        """Instantiate class to allocate payments.

        Args:
            X (np.ndarray): Array of raw feature variables.
            y (np.ndarray): The additive gaussian noise added to the target variable.
            forecast_model (Model): Model used to calculate weights and evalaute objective.
            dummy_agents (Tuple[int]): Indicies of dummy variables in X.
            central_agents (Tuple[int]): Indicies of variables in X associated with central agents.
            support_agents (Tuple[int]): Indicies of variables in X associated with support agents.
            polynomial_order (int): If the polynomial order is greater than 1, variable iteractions
                may exist between the central agents variables and the support agent variables and
                will impact the allocation/payments received. Note, in this case, the allocation of
                the support agents may not necessesarily sum to 1.
        """
        self.X = X
        self.y = y
        self.forecast_model = forecast_model
        self.dummy_agents = dummy_agents
        self.central_agents = central_agents
        self.support_agents = support_agents
        self.polynomial_order = polynomial_order
        self.all_agents = self.central_agents + self.support_agents

        self._sort_columns()
        self._reset_agent_indicies()

        # If the polynomial order is greater than 1, variable iteractions may exist
        # between the central agents variables and the support agent variables and
        # will impact the allocation/payments received. Note, in this case, the allocation
        # of the support agents may not necessesarily sum to 1.
        self.agents_to_evaluate = self.support_agents
        if self.polynomial_order > 1:
            self.agents_to_evaluate += self.central_agents
        self.baseline_agents = list(self.dummy_agents) + list(
            set(self.central_agents).difference(set(self.agents_to_evaluate))
        )

    def _sort_columns(self):
        agents = self.dummy_agents + self.all_agents
        combinations = len(self._combinations(self.all_agents, 2, self.polynomial_order))
        interactions = np.arange(combinations) + len(self.dummy_agents) + len(self.all_agents)
        self.X = self.X[:, agents + tuple(interactions)]

    def _reset_agent_indicies(self):
        self.dummy_agents = tuple(np.arange(len(self.dummy_agents)))
        self.central_agents = tuple(np.arange(len(self.central_agents)) + len(self.dummy_agents))
        self.support_agents = tuple(
            np.arange(len(self.support_agents)) + len(self.dummy_agents) + len(self.central_agents)
        )

    def _coalition_weights(
        self, X: np.ndarray, y: np.ndarray, indicies: List[int], all_weights: np.ndarray = None
    ) -> np.ndarray:
        if all_weights is None:
            return self.forecast_model.calculate_weights(X[:, np.sort(indicies)], y)
        return all_weights[np.sort(indicies)]

    def _objective(
        self, X: np.ndarray, y: np.ndarray, indicies: List[int], all_weights: np.ndarray = None
    ) -> float:
        weights = self._coalition_weights(X, y, indicies, all_weights=all_weights)
        return self.forecast_model.calculate_loss(y, X[:, np.sort(indicies)] @ weights)

    def _combinations(self, seq: Sequence[int], min: int, max: int, replace=True) -> List:
        iterator = itertools.combinations_with_replacement if replace else itertools.combinations
        return list(itertools.chain.from_iterable((iterator(seq, i) for i in range(min, max + 1))))

    def _marginal_contribution(
        self, X: np.ndarray, y: np.ndarray, excl: Sequence[int], incl: Sequence[int], **kwargs
    ) -> float:
        return self._objective(X, y, excl, **kwargs) - self._objective(X, y, incl, **kwargs)

    def _marginal_contribution_weight(self, *args) -> float:
        raise NotImplementedError

    def _weighted_average_marginal_contributions(
        self, X: np.ndarray, y: np.ndarray, **kwargs
    ) -> np.ndarray:
        agent_combinations = self._combinations(self.agents_to_evaluate, 1, self.polynomial_order)

        marginal_contributions = []
        for agent in self.agents_to_evaluate:
            remaining_agents = [a for a in self.agents_to_evaluate if a != agent]
            remaining_agent_combinations = self._combinations(
                remaining_agents, 0, len(remaining_agents), replace=False
            )

            agent_marginal_contributions = []
            for combination in remaining_agent_combinations:

                def coalition(include_evaluating_agent: bool):
                    if include_evaluating_agent:
                        agents = {agent}.union(set(combination))
                    else:
                        agents = set(combination)
                    return self.baseline_agents + [
                        agent_combinations.index(agent_combination) + len(self.baseline_agents)
                        for agent_combination in agent_combinations
                        if set(agent_combination).issubset(agents)
                    ]

                agent_marginal_contributions.append(
                    self._marginal_contribution_weight(len(combination))
                    * self._marginal_contribution(X, y, coalition(False), coalition(True), **kwargs)
                )

            marginal_contributions.append(agent_marginal_contributions)

        return np.array(marginal_contributions).sum(axis=1)

    def _grand_coalition_contribution(self, X: np.ndarray, y: np.ndarray, **kwargs) -> float:
        return self._marginal_contribution(
            X, y, self.baseline_agents, np.arange(X.shape[1]), **kwargs
        )

    def _allocation(
        self, X: np.ndarray, y: np.ndarray, marginal_contributions: np.ndarray, **kwargs
    ) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            grand_coalition_contribution = self._grand_coalition_contribution(X, y, **kwargs)
            allocation = np.true_divide(marginal_contributions, grand_coalition_contribution)
            allocation[allocation == np.inf] = 0
        return np.nan_to_num(allocation)

    def _batch(
        self,
        X: np.ndarray,
        y: np.ndarray,
        willingness_to_pay: float,
        all_weights: np.ndarray = None,
    ) -> Tuple[np.ndarray]:
        contributions = self._weighted_average_marginal_contributions(X, y, all_weights=all_weights)
        return {
            "allocations": self._allocation(X, y, contributions, all_weights=all_weights),
            "payments": contributions * len(X) * willingness_to_pay,
            "weights": self.forecast_model.calculate_weights(X, y)
            if all_weights is None
            else all_weights,
        }

    def _online(self, willingness_to_pay: float, state: StateTracker) -> Tuple[np.ndarray]:
        num_samples = len(self.X)
        marginal_contributions = np.zeros((num_samples, len(self.agents_to_evaluate)))
        allocations = np.zeros((num_samples, len(self.agents_to_evaluate)))

        for i in tqdm(range(num_samples)):
            state.log_data(self.X[i, :].reshape(1, -1), self.y[i])
            state.update_memory()

            if i >= state.contributions_burn_in:
                marginal_contributions[i, :] = (
                    self._weighted_average_marginal_contributions(
                        state.X,
                        state.y,
                        all_weights=state.current_weights,
                    )
                    * (1 - state.forgetting_factor)
                    + state.forgetting_factor * marginal_contributions[i - 1, :]
                )
                allocations[i, :] = self._allocation(
                    state.X,
                    state.y,
                    marginal_contributions[i, :],
                    all_weights=state.current_weights,
                )

            if i > state.weights_burn_in:
                state.update_weights()

        return {
            "allocations": allocations,
            "payments": marginal_contributions * willingness_to_pay,
            "weights": state._estimated_weights,
        }

    def _split_data(self, test_frac: float, train: bool):
        idx = int(self.X.shape[0] * (1 - test_frac))
        if train:
            return self.X[:idx], self.y[:idx]
        return self.X[idx:], self.y[idx:]

    def run_batch_market(
        self,
        test_frac: float,
        willingness_to_pay: float,
        train: bool = True,
        all_weights: np.ndarray = None,
    ) -> Dict:
        if not train and all_weights is None:
            raise ValueError("Weights must be provided for out of sample run")
        X, y = self._split_data(test_frac, train)
        return self._batch(X, y, willingness_to_pay)

    def run_online_market(
        self, state: StateTracker, willingness_to_pay: float, train: bool = True
    ) -> Dict:
        ### TODO: Combine in-sample and out-of-sample online markets so we only have to
        ### TODO: run this loop once.
        return self._online(willingness_to_pay, state)


class Shapley(SemivaluePolicy):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        forecast_model: Model,
        dummy_agents: Tuple[int],
        central_agents: Tuple[int],
        support_agents: Tuple[int],
        polynomial_order: int = 1,
    ):
        super().__init__(
            X,
            y,
            forecast_model,
            dummy_agents,
            central_agents,
            support_agents,
            polynomial_order=polynomial_order,
        )

    def _marginal_contribution_weight(self, size: int) -> float:
        return (
            np.math.factorial(size)
            * np.math.factorial(len(self.agents_to_evaluate) - size - 1)
            / np.math.factorial(len(self.agents_to_evaluate))
        )
