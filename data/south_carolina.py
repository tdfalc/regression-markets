from pathlib import Path

import pandas as pd
import numpy as np


def build_data(fname: Path, central_agent: int, num_samples: int, max_lags: int = 2):
    """Build market data for specified central agent using the South Carolina wind power outputs."""

    data = pd.read_csv(fname, header=None, index_col=0).to_numpy()
    data = data / data.max(axis=0)

    def time_shift(array: np.ndarray, lag: int):
        result = np.empty_like(array)
        if lag > 0:
            result[:lag] = np.nan
            result[lag:] = array[:-lag]
        elif lag < 0:
            result[lag:] = np.nan
            result[:lag] = array[-lag:]
        else:
            result[:] = array
        return result

    all_agents = np.arange(data.shape[1])
    support_agents = all_agents[all_agents != central_agent]
    central_agent_lags = np.vstack([time_shift(data[:, central_agent], i) for i in (1, 2)]).T
    support_agent_lags = np.vstack([time_shift(data[:, a], 1) for a in support_agents]).T

    X = np.hstack([central_agent_lags, support_agent_lags])[max_lags:]
    X = np.insert(X, 0, 1, axis=1)

    y = data[max_lags:, central_agent]

    return X[:num_samples], y[:num_samples]
