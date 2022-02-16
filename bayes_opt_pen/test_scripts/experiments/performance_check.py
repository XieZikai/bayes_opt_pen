from botorch.models.gpytorch import GPyTorchModel
from gpytorch.distributions import MultivariateNormal
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.models import ExactGP
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior


class SimpleCustomGP(ExactGP, GPyTorchModel):

    _num_outputs = 1  # to inform GPyTorchModel API

    def __init__(self, train_X, train_Y):
        # squeeze output dim before passing train_Y to ExactGP
        super().__init__(train_X, train_Y.squeeze(-1), GaussianLikelihood())
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(
            base_kernel=RBFKernel(ard_num_dims=train_X.shape[-1]),
        )
        self.to(train_X)  # make sure we're on the right device/dtype

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class LinearMeanGP(ExactGP, GPyTorchModel):
    _num_outputs = 1  # to inform GPyTorchModel API

    def __init__(self, train_X, train_Y):
        # squeeze output dim before passing train_Y to ExactGP
        super().__init__(train_X, train_Y.squeeze(-1), GaussianLikelihood())
        self.mean_module = LinearMean(2)
        self.covar_module = ScaleKernel(
            base_kernel=RBFKernel(ard_num_dims=train_X.shape[-1]),
        )
        self.to(train_X)  # make sure we're on the right device/dtype

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


from botorch.fit import fit_gpytorch_model


def _get_and_fit_simple_custom_gp(Xs, Ys, **kwargs):
    model = SimpleCustomGP(Xs[0], Ys[0])
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)
    return model


def _get_and_fit_linear_mean_gp(Xs, Ys, **kwargs):
    model = LinearMeanGP(Xs[0], Ys[0])
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)
    return model


import random
import numpy as np


def branin(parameterization, *args):
    x1, x2 = parameterization["x1"], parameterization["x2"]
    y = (x2 - 5.1 / (4 * np.pi ** 2) * x1 ** 2 + 5 * x1 / np.pi - 6) ** 2 - x1 - x2
    y += 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1) + 10
    # let's add some synthetic observation noise
    y += random.normalvariate(0, 0.1)
    return {"branin": (y, 0.0)}


from ax import ParameterType, RangeParameter, SearchSpace

search_space = SearchSpace(
    parameters=[
        RangeParameter(
            name="x1", parameter_type=ParameterType.FLOAT, lower=-5, upper=10
        ),
        RangeParameter(
            name="x2", parameter_type=ParameterType.FLOAT, lower=0, upper=15
        ),
    ]
)

from ax import SimpleExperiment

from ax.modelbridge import get_sobol
from ax.modelbridge.factory import get_botorch


flag = 0
total = 100

for _ in range(100):
    try:
        exp = SimpleExperiment(
            name="test_branin",
            search_space=search_space,
            evaluation_function=branin,
            objective_name="branin",
            minimize=True,
        )

        exp_linear = SimpleExperiment(
            name="test_branin",
            search_space=search_space,
            evaluation_function=branin,
            objective_name="branin",
            minimize=True,
        )
        sobol = get_sobol(exp.search_space)
        exp.new_batch_trial(generator_run=sobol.gen(5))

        sobol = get_sobol(exp_linear.search_space)
        exp_linear.new_batch_trial(generator_run=sobol.gen(5))

        min1 = 500
        min2 = 500

        iteration = np.random.randint(10, 40)

        for i in range(iteration):
            print('=========')
            print(f"None-linear: Running optimization batch {i + 1}/50...")
            model = get_botorch(
                experiment=exp,
                data=exp.eval(),
                search_space=exp.search_space,
                model_constructor=_get_and_fit_simple_custom_gp,
            )
            batch = exp.new_trial(generator_run=model.gen(1))
            data = exp.eval_trial(batch)
            print('Arms: ', batch.arms)
            result = branin(batch.arm.parameters)['branin'][0]
            print('Result: ', result)
            min1 = min(min1, result)
            print('-----------')

            print(f"Linear: Running optimization batch {i + 1}/50...")
            model1 = get_botorch(
                experiment=exp_linear,
                data=exp_linear.eval(),
                search_space=exp_linear.search_space,
                model_constructor=_get_and_fit_linear_mean_gp,
            )
            batch1 = exp_linear.new_trial(generator_run=model.gen(1))
            data = exp_linear.eval_trial(batch1)
            print('Arms: ', batch1.arms)
            result = branin(batch1.arm.parameters)['branin'][0]
            print('Result: ', result)
            min2 = min(min2, result)

        print("Done!")
        print('Min1: ', min1)
        print('Min2: ', min2)
        if min2 < min1:
            flag += 1
    except:
        total -= 1

print('flag: ', flag, ' out of ', total)
