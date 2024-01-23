# Regression Markets

## v2 - Capricious data streams 

Python implemention of [Towards Regression Markets with Capricious Data Streams](https://arxiv.org/abs/2310.14992). 

* Transition to Gaussian Process regression analyses.
* Currently only explores out-of-sample market.

If you find this useful in your work, we kindly request that you cite the following publication:

```
@misc{falconer2023bayesian,
      title={Towards Regression Markets with Capricious Data Streams}, 
      author={Thomas Falconer and Jalal Kazempour and Pierre Pinson},
      year={2023},
      eprint={2310.14992},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## v1 - Towards Bayesian regression markets
 
Python implemention of [Bayesian Regression Markets](https://arxiv.org/abs/2310.14992). 

* Transition to Bayesian regression analyses.
* Allocation policies based on both log-likelihood and KL divergence.

If you find this useful in your work, we kindly request that you cite the following publication(s):

```
@misc{falconer2023bayesian,
      title={Bayesian Regression Markets}, 
      author={Thomas Falconer and Jalal Kazempour and Pierre Pinson},
      year={2023},
      eprint={2310.14992},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

```
@misc{falconer2023incentivizing,
      title={Replication-Robust Analytics Markets}, 
      author={Thomas Falconer and Jalal Kazempour and Pierre Pinson},
      year={2023},
      eprint={2310.06000},
      archivePrefix={arXiv},
      primaryClass={econ.GN}
}
```

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