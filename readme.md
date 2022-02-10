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



## Usage

### bayes_opt

```
```

### botorch

Below is a botorch sample of using acquisition functions with penalty term:

Define problem:
```python
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import UpperConfidenceBound
from bayes_opt_pen.gpytorch_utils.acq import PenalisedUpperConfidenceBound


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

Use penalised acquisition function and follow the optimization procedure:
```python
UCB_pen = PenalisedUpperConfidenceBound(gp, beta=0.1)
UCB = UpperConfidenceBound(gp, beta=0.1)

from botorch.optim import optimize_acqf

bounds = torch.stack([torch.zeros(2), torch.ones(2)])
iter = 100

for i in range(iter):
    candidate, acq_value = optimize_acqf(
        UCB, bounds=bounds, q=1, num_restarts=5, raw_samples=20,
    )
    print(candidate, 1-torch.norm(candidate-0.5))

```