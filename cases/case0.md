# Case 0
Preliminary illustration of the regression market setup on a set of simulation studies, as well as a real-world forecasting example.

## Simulation 1
Batch learning, stationary process, plain linear regression with Least-Squares estimation. Data is generated for a central agent (agent 1) and two support agents. The central agent owns $x_{1}$, agent two owns $x_{2}$ and agent three owns both $x_{3}$ and $x_{4}$.

The features are sampled from a Gaussian distribution, $x_{j} ∼ \mathcal{N}(0, 1)$ with additive noise $\epsilon ∼ \mathcal{N}(0, 0.3)$. The regression task chosen by the central agent (which is well specifed in view of the true data and posted on the analytics platform is a model of the form:

$$
y = w_{0} + w_{1} x_{1, t} + w_{2} x_{2, t} + w_{3} x_{3, t} + w_{4} x_{4, t} + ϵ_{t}
$$

First, the value of the features of the support agents are assessed with a batch leave-one-out policy. The central agent has a willingness to pay of 0.1 euro/datapoint/unit improvement in MSE.

Then we assess the value of the features of the support agents with a batch Shapley policy. The central agent again has a willingness to pay of 0.1 euro/datapoint/percentpoint improvement in MSE.

## Simulation 2
Next we generalize to a second order polynomial regression, with the same number of agents and a quadratic loss. The central agent owns $x_{1}$, agent two owns $x_{2}$ and agent three owns $x_{3}$.

The features are sampled from a Gaussian distribution, $x_{j} ∼ \mathcal{N}(0, 1)$ with additive noise $\epsilon ∼ N(0, 0.3)$. The regression task posted on the analytics platform is a model of the form:

$$
y = w_{0} + w_{1} x_{1, t} + w_{2} x_{2, t} + w_{3} x_{3, t} + w_{4} x_{1, t}^{2} +
    w_{5} x_{1, t} x_{2, t} + \\ w_{6} x_{1, t} x_{3, t} + w_{7} x_{2, t}^{2} +
    w_{8} x_{2, t} x_{3, t} + w_{9} x_{3, t}^{2} + ϵ_{t}
$$

We assess the value of the features of the support agents with a batch Shapley approach. The central agent has a willingness to pay of 0.1 euro/datapoint/percentpoint improvement in MSE.

## Simulation 3
Here, the central agent wants to learn an Auto-Regressive with eXogenous input (ARX) model
with a quantile loss function (with nominal level $\tau$), based on a one timestep lag of the
target variable and one timestep lags of the features from the support agents. Agent two
owns the first feature, and agent three owns the last two features.

The features are sampled from a Gaussian distribution, $x_{j} ∼ \mathcal{N}(0, 1)$ with additive noise $\epsilon ∼ \mathcal{N}(0, 0.3)$. The regression task posted on the analytics platform is a model of the form:

$$
y = w_{0} + w_{1} y_{t-1} + w_{2} x_{1, t-1} + w_{3} x_{2, t-1} + w_{4} x_{3, t-1} + ϵ_{t}
$$

We assess the value of the features of the support agents with a batch Shapley approach.
The central agent has a willingness to pay of 0.1 euro/datapoint/percentpoint improvement
in MSE.

## Simulation 4
The central agent now aims to use online learning with a quadratic loss for the ARX model.
The major diference here is that the parameters vary in time.

The features are sampled from a Gaussian distribution, $x_{j} ∼ \mathcal{N}(0, 1)$ with additive nosie $\mathcal{epsilon} ∼ N(0, 0.3)$. The regression task posted on the analytics platform is a model of the form:

$$
y = w_{0, t} + w_{1, t} y_{t-1} + w_{2, t} x_{1, t-1} + w_{3, t} x_{4, t-1} + 
    w_{5, t} x_{3, t-1} + ϵ_{t}
$$

Since the online regression market relies on an online learning component, the parameters
are tracked in time, and with the payments varying accordingly. We assess the value of
the features of the support agents with a batch Shapley approach. The central agent has a
willingness to pay of 0.1 euro/datapoint/percentpoint improvement in MSE.

## Simulation 5
For this last synthetic simulation case, let us consider a linear quantile regression model, hence with a central agent aiming to perform online learning with a smooth quantile loss function.

The central agent posts the task on the analytics platform, with a forgetting factor of
0.999. The parameter α of the smooth quantile loss function is set to $\alpha = 0.2$.

The regression task posted on the analytics platform relies on a model of the form:

$$
y = w_{0, t} + w_{1, t} x_{1, t} + w_{2, t} x_{2, t} + w_{3, t} x_{3, t} +
w_{4, t} x_{4, t} ϵ_{t}
$$

The features $x_{1}$, $x_{2}$ and $x_{3}$ are sampled from a Gaussian distribution, $x_{j} ∼ \mathcal{N}(0, 1)$ and feature $x_{4}$ is sampled from a Uniform distributio, $x_{4} ~ \mathcal{U}(0.5, 1.5)$ with additive noise term $\mathcal{eps} ∼ N(0, 0.3)$. The standard deviation of the noise is then scaled by $w_{4} x_{4}$.

That means that $x_{1}$, $x_{2}$ and $x_{3}$ are important features to model the mean (or median) of y, whilst $x_{4}$ will have an increased improtance when aiming to model quantiles that are further away from the median (i.e. with nominal levels tending towards 0 and 1).
        
## Simulation 6
In this real-world cast study, we here use the South Carolina dataset extracted from the [Wind Toolkit](https://www.nrel.gov/hpc/eagle-wind-dataset.html) with hourly data for 9 wind farms within a 150 km radius.

The first 10000 points are used for a batch regression market, and the following 10000 points are used for the corresponding out-of-sample market. The in-sample willingness to pay is 0.5 euro per percent point improvement and per data point, then 1.5 for out-of-sample.

## Simulation 7
We again use the South Carolina dataset extracted from the Wind Toolkit, with hourly data for 9 wind farms within a 150 km radius, this time for an online setup.

The first 500 points are used to kickstart the online regression market (burn-in period), and after that, at each time step, the online and out-of-sample regression markets come one after the other. The in-sample willingness to pay is of 0.2 euro per percent point improvement and per data point, then 0.8 out-of-sample.