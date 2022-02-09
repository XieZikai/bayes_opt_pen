from abc import ABC

from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.models.model import Model
from typing import Dict, Optional, Tuple, Union
from botorch.acquisition.objective import ScalarizedObjective
from botorch.utils.transforms import convert_to_target_pre_hook, t_batch_mode_transform

import torch
from torch.distributions import Normal
from torch import Tensor


"""
This file contains acquisition functions with penalty term.
All acquisition functions requires datapoint-like prior knowledge provided by GPR model: gp.prior_point_list.
"""


class PenalisedAnalyticAcquisitionFunction(AnalyticAcquisitionFunction, ABC):
    def __init__(
        self, model: Model, objective: Optional[ScalarizedObjective] = None
    ) -> None:
        r"""Base constructor for analytic acquisition functions.

        Args:
            model: A fitted single-outcome model with datapoint-like prior knowledge 'prior_point_list'.
            objective: A ScalarizedObjective (optional).
        """
        # assertion for prior knowledge
        assert hasattr(model, 'prior_point_list'), 'Model must provide datapoint-like prior knowledge!'
        for prior in model.prior_point_list:
            if isinstance(model.train_inputs, tuple):
                assert model.train_inputs[0].shape[-1] == prior.shape[0], 'Prior must be with the same shape as input!'

        self.prior_pointer = 0
        self.constant_multiplier = None

        # initialize learning rate
        self.lr = torch.ones(len(model.prior_point_list))

        self.chosen_index = 0

        super().__init__(model=model, objective=objective)


class PenalisedExpectedImprovement(PenalisedAnalyticAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        best_f: Union[float, Tensor],
        objective: Optional[ScalarizedObjective] = None,
        maximize: bool = True,
    ) -> None:
        r"""Single-outcome Expected Improvement (analytic).

        Args:
            model: A fitted single-outcome model with datapoint-like prior knowledge 'prior_point_list'.
            best_f: Either a scalar or a `b`-dim Tensor (batch mode) representing
                the best function value observed so far (assumed noiseless).
            objective: A ScalarizedObjective (optional).
            maximize: If True, consider the problem a maximization problem.
        """
        super().__init__(model=model, objective=objective)
        self.maximize = maximize
        if not torch.is_tensor(best_f):
            best_f = torch.tensor(best_f)
        self.register_buffer("best_f", best_f)

    @t_batch_mode_transform(expected_q=1, assert_output_shape=False)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate Expected Improvement on the candidate set X.

        Args:
            X: A `b1 x ... bk x 1 x d`-dim batched tensor of `d`-dim design points.
                Expected Improvement is computed for each point individually,
                i.e., what is considered are the marginal posteriors, not the
                joint.

        Returns:
            A `b1 x ... bk`-dim tensor of Expected Improvement values at the
            given design points `X`.
        """
        self.best_f = self.best_f.to(X)
        posterior = self._get_posterior(X=X)
        mean = posterior.mean
        # deal with batch evaluation and broadcasting
        view_shape = mean.shape[:-2] if mean.dim() >= X.dim() else X.shape[:-2]
        mean = mean.view(view_shape)
        sigma = posterior.variance.clamp_min(1e-9).sqrt().view(view_shape)
        u = (mean - self.best_f.expand_as(mean)) / sigma
        if not self.maximize:
            u = -u
        normal = Normal(torch.zeros_like(u), torch.ones_like(u))
        ucdf = normal.cdf(u)
        updf = torch.exp(normal.log_prob(u))
        ei = sigma * (updf + u * ucdf)
        return ei