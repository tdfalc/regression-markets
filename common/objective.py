import numpy as np


def expected_kl_divergence_univariate_normal(
    prior_mean: np.ndarray,
    prior_variance: np.ndarray,
    posterior_mean: np.ndarray,
    posterior_variance: np.ndarray,
):
    mean_delta = posterior_mean.flatten() - prior_mean.flatten()
    prior_sdev = prior_variance.flatten() ** 0.5
    posterior_sdev = posterior_variance.flatten() ** 0.5
    return (
        np.log(prior_sdev / posterior_sdev)
        + (posterior_sdev**2 + mean_delta**2) / (2 * prior_sdev**2)
        - 0.5
    ).mean()
