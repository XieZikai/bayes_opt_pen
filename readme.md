https://github.com/RichardLitt/standard-readme/blob/master/example-readmes/maximal-readme.md

# Penalised Acquisition Functions


This package contains acquisition functions with penalty terms for Bayesian optimization, 
which is a novel method of combining datapoint-like prior knowledge into BO process.

## Table of Contents

 - [Background](#background)
 - [Install](#install)
 - [Usage](#usage)


## Background



## Install

Haven't registered yet.

```shell
python setup.py build
python setup.py install
```

## Usage

### bayes_opt

Below is a `bayes_opt` sample of using acquisition functions with penalty term:

Define problem:
```python
import numpy as np

def black_box_function(x, y):
    return -x ** 2 - (y - 1) ** 2 + 1
```

Define prior point list:
```python
prior_point_list = [2.5, 3.5]
```

Calling optimizer:
```python
from bayes_opt_pen.bayesian_optimization import BayesianOptimization

# Bounded region of parameter space
pbounds = {'x': (2, 4), 'y': (-3, 3)}

optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=pbounds,
    random_state=1,
    prior_point_list=prior_point_list
)
```

```python
optimizer.maximize(
    init_points=2,
    n_iter=3,
)
```

### botorch

Below is a `botorch` sample of using acquisition functions with penalty term:

Define problem:
```python
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

train_X = torch.rand(10, 2)
Y = 1 - torch.norm(train_X - 0.5, dim=-1, keepdim=True)
Y = Y + 0.1 * torch.randn_like(Y)  # add some noise
train_Y = standardize(Y)

gp = SingleTaskGP(train_X, train_Y)
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_model(mll)

gp = SingleTaskGP(train_X, train_Y)
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_model(mll)
```

Adding prior knowledge: directly define `prior_point_list` for gpytorch model:
```python
prior_knowledge = [torch.rand(2), torch.rand(2)]
gp.prior_point_list = prior_knowledge
```

Using penalised acquisition function and following the optimization procedure:
```python
from bayes_opt_pen.gpytorch_utils.acq import PenalisedUpperConfidenceBound

UCB_pen = PenalisedUpperConfidenceBound(gp, beta=0.1)

from botorch.optim import optimize_acqf

bounds = torch.stack([torch.zeros(2), torch.ones(2)])
iter = 100

for i in range(iter):
    candidate, acq_value = optimize_acqf(
        UCB, bounds=bounds, q=1, num_restarts=5, raw_samples=20,
    )
    print(candidate, 1-torch.norm(candidate-0.5))

```