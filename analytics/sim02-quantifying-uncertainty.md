# Simulation 02 - Quantifying Uncertainty

## 1. Description
For the simple linear regression case as described in `sim01-visualizing-uncertainty`, we perform Bayesian inference to estimate the model parameters. Predictions can then be made using the posterior predictive distribution. Since this model has only two parameters, `w0` and `w1`, we can visualize the learning process.

## 2. Results
The leftmost column presents the progressive evolution of the posterior density as the sample size increases. In the middle column, random weight samples are drawn from the posterior distribution, and the associated regression lines are plotted in red. The black dashed line represents the regression line determined by the *true* parameters, while the noisy training data is represented by black dots. In the rightmost column, both the mean and standard deviation of the posterior predictive distribution are displayed alongside the true model and the training data

Observing the figures, we see that the posterior density in the first column becomes more tightly concentrated as the dataset size increases, indicating a decrease in sample variance in the second column and a subsequent reduction in prediction uncertainty as demonstrated in the third column. Additionally, we can clearly see that prediction uncertainty is more pronounced in regions with fewer observations.

![](./docs/sim02-quantifying-uncertainty/bayesian_updates.png)
