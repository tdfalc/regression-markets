# Regression Markets

## v1 - Towards Bayesian regression markets
 
Python implemention of [Bayesian Regression Markets](). 

* Transition to Bayesian regression analyses.
* Allocation policies based on both log-likelihood and KL divergence.

## v0 - Introducing regression markets
 
Python implemention of [Regression markets and application to energy forecasting](https://link.springer.com/article/10.1007/s11750-022-00631-7). 

* Generalised implementation for semivalue-based attribution policies.
* Batch and online (in-sample/out-of-sample) experiments with linear/quantile regression models.
* Simulation cases and results stored in `./cases` with descriptions for each case presented on the corresponding markdown file.
  
> `case0` Simulations (batch, online, in-sample) using synthetic data, as well as real-world forecasting case-studies (batch, online, in-sample, out-of-sample) for South Carolina wind farm power output.


If you find this useful in your work, we kindly request that you cite the following publication:

```
@article{pinson2022regression,
    title={Regression markets and application to energy forecasting},
    author={Pierre Pinson and Liyang Han and Jalal Kazempour},
    year={2022},
    doi={10.1007/s11750-022-00631-7},
    journal={TOP},
    publisher={Springer},
}
```