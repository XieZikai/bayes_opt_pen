from abc import ABC

from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.models.model import Model
from typing import Optional, Union
from botorch.acquisition.objective import ScalarizedObjective
from botorch.utils.transforms import t_batch_mode_transform

import torch
from torch.distributions import Normal
from torch import Tensor

from bayes_opt_pen.gpytorch_utils.utils import del_tensor_element

"""
This file contains acquisition functions with penalty term.
All acquisition functions requires datapoint-like prior knowledge provided by GPR model: gp.prior_point_list.
"""


class PenalisedAnalyticAcquisitionFunction(AnalyticAcquisitionFunction, ABC):
    def __init__(
            self, model: Model, objective: Optional[ScalarizedObjective] = None,
            lr_decay=0.1
    ) -> None:
        r"""Base constructor for analytic acquisition functions.

        Args:
            model: A fitted single-outcome model with datapoint-like prior knowledge 'prior_point_list'.
            objective: A ScalarizedObjective (optional).
            lr_decay: Learning rate decay of penalty term.
        """
        # assertion for prior knowledge
        assert hasattr(model, 'prior_point_list'), 'Model must provide datapoint-like prior knowledge!'
        for prior in model.prior_point_list:
            if isinstance(model.train_inputs, tuple):
                assert model.train_inputs[0].shape[-1] == prior.shape[0], 'Prior must be with the same shape as input!'
        self.prior_point_list = model.prior_point_list

        self.prior_pointer = 0
        self.constant_multiplier = None

        # initialize learning rate
        self.lr = torch.ones(len(model.prior_point_list))
        self._lr_decay = lr_decay
        self.chosen_index = 0

        super().__init__(model=model, objective=objective)

    def compute_penalty_term(self, X: Tensor, acq_score):

        if len(self.lr) == 0:
            return 0

        chosen_index = self.chosen_index
        self.prior_pointer = chosen_index
        lr = self.lr[chosen_index]
        dist = torch.norm(X - self.prior_point_list[chosen_index])

        # Adding multiplier to make sure the penalty term is at the same scale as the UCB value
        if self.constant_multiplier is None:
            if isinstance(acq_score, float):
                self.constant_multiplier = acq_score / (lr * torch.sqrt(dist) * 2)
            elif isinstance(acq_score, torch.Tensor) or isinstance(acq_score, list):
                self.constant_multiplier = acq_score[0] / (lr * torch.sqrt(dist) * 2)

        penalty_term = lr * torch.sqrt(dist) * self.constant_multiplier * 2
        return penalty_term

    def delete_prior(self):
        if len(self.lr) > 0:
            self.lr = del_tensor_element(self.lr, self.prior_pointer)
            del(self.prior_point_list[self.prior_pointer])
            self.prior_pointer -= 1

    def update_params(self):
        if len(self.lr) > 0:
            self.lr[self.prior_pointer] *= self._lr_decay
            self.prior_pointer = (self.prior_pointer + 1) % len(self.prior_point_list)


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

        penalty_term = self.compute_penalty_term(X, ei)
        return ei - penalty_term


class PenalisedProbabilityOfImprovement(PenalisedAnalyticAcquisitionFunction):

    def __init__(
        self,
        model: Model,
        best_f: Union[float, Tensor],
        objective: Optional[ScalarizedObjective] = None,
        maximize: bool = True,
    ) -> None:
        r"""Single-outcome analytic Probability of Improvement.

        Args:
            model: A fitted single-outcome model.
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

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the Probability of Improvement on the candidate set X.

        Args:
            X: A `(b) x 1 x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.

        Returns:
            A `(b)`-dim tensor of Probability of Improvement values at the given
            design points `X`.
        """
        self.best_f = self.best_f.to(X)
        posterior = self._get_posterior(X=X)
        mean, sigma = posterior.mean, posterior.variance.sqrt()
        batch_shape = X.shape[:-2]
        mean = posterior.mean.view(batch_shape)
        sigma = posterior.variance.sqrt().clamp_min(1e-9).view(batch_shape)
        u = (mean - self.best_f.expand_as(mean)) / sigma
        if not self.maximize:
            u = -u
        normal = Normal(torch.zeros_like(u), torch.ones_like(u))
        poi = normal.cdf(u)

        penalty_term = self.compute_penalty_term(X, poi)
        return poi - penalty_term


class PenalisedUpperConfidenceBound(PenalisedAnalyticAcquisitionFunction):

    def __init__(
        self,
        model: Model,
        beta: Union[float, Tensor],
        objective: Optional[ScalarizedObjective] = None,
        maximize: bool = True,
    ) -> None:
        r"""Single-outcome Upper Confidence Bound.

        Args:
            model: A fitted single-outcome GP model (must be in batch mode if
                candidate sets X will be)
            beta: Either a scalar or a one-dim tensor with `b` elements (batch mode)
                representing the trade-off parameter between mean and covariance
            objective: A ScalarizedObjective (optional).
            maximize: If True, consider the problem a maximization problem.
        """
        super().__init__(model=model, objective=objective)
        self.maximize = maximize
        if not torch.is_tensor(beta):
            beta = torch.tensor(beta)
        self.register_buffer("beta", beta)

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the Upper Confidence Bound on the candidate set X.

        Args:
            X: A `(b) x 1 x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.

        Returns:
            A `(b)`-dim Tensor of Upper Confidence Bound values at the given
            design points `X`.
        """
        self.beta = self.beta.to(X)
        posterior = self._get_posterior(X=X)
        batch_shape = X.shape[:-2]
        mean = posterior.mean.view(batch_shape)
        variance = posterior.variance.view(batch_shape)
        delta = (self.beta.expand_as(mean) * variance).sqrt()
        if self.maximize:
            ucb = mean + delta
        else:
            ucb = -mean + delta

        penalty_term = self.compute_penalty_term(X, ucb)
        return ucb - penalty_term

