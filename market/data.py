import numpy as np

from market.basis import add_polynomial_features


class MarketData:
    def __init__(
        self,
        dummy_feature: np.ndarray,
        central_agent_features: np.ndarray,
        support_agent_features: np.ndarray,
        target_signal: np.ndarray,
    ):
        """Instantiate `MarketData`.
        Args:
            dummy_feature (np.ndarray): Dummy feature (typically a vector ofones).
            central_agent_features (np.ndarray): Observations for features owned
                by the central_agent.
            support_agent_features (np.ndarray): Observations for features owned
                by the support_agents.
            target_signal (np.ndarray): Ground truth values for the target signal.
        """

        self.dummy_feature = dummy_feature
        self.central_agent_features = central_agent_features
        self.support_agent_features = support_agent_features
        self.target_signal = target_signal

        self.num_central_agent_features = self.central_agent_features.shape[1]
        self.num_support_agent_features = self.support_agent_features.shape[1]

        self.X = self._build_design_matrix()
        self.y = target_signal

        self._set_agent_indices()

    def _build_design_matrix(self):
        market_features = np.hstack(
            [self.central_agent_features, self.support_agent_features]
        )
        return np.hstack([self.dummy_feature, market_features])

    def _set_agent_indices(self):
        central_agents = set(np.arange(self.num_central_agent_features) + 1)
        support_agents = set(
            np.arange(self.num_support_agent_features) + max(central_agents) + 1
        )
        self.active_agents = support_agents
        self.baseline_agents = set([0]).union(
            central_agents.difference(self.active_agents)
        )


class BatchData(MarketData):
    def __init__(
        self,
        dummy_feature: np.ndarray,
        central_agent_features: np.ndarray,
        support_agent_features: np.ndarray,
        target_signal: np.ndarray,
        test_frac: float = 0,
    ):
        super().__init__(
            dummy_feature=dummy_feature,
            central_agent_features=central_agent_features,
            support_agent_features=support_agent_features,
            target_signal=target_signal,
        )

        self.test_frac = test_frac
        self._split_data()

    def _split_data(self):
        index = int(self.X.shape[0] * (1 - self.test_frac))
        self.X_train, self.X_test = self.X[:index], self.X[index:]
        self.y_train, self.y_test = self.y[:index], self.y[index:]

    @property
    def train_data(self):
        return self.X_train, self.y_train

    @property
    def test_data(self):
        return self.X_test, self.y_test
