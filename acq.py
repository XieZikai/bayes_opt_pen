import numpy as np
import warnings
from scipy.stats import norm


class ModifiedAcq(object):

    def __init__(self, kind, kappa, xi, prior_point_list, kappa_decay=1, kappa_decay_delay=0, lr_decay=0.1):
        self.kappa = kappa
        self._kappa_decay = kappa_decay
        self._kappa_decay_delay = kappa_decay_delay

        self.xi = xi
        self.constant_multiplier = None

        self._iters_counter = 0

        self.kind = kind
        self.prior_point_list = prior_point_list
        self.lr = np.ones(len(prior_point_list))

        self._lr_decay = lr_decay

        self.prior_pointer = None

        self.chosen_index = 0

    def utility(self, x, gp, y_max):
        if self.kind == 'ucb':
            assert self.prior_point_list is not None, 'Prior point list should not be none'
            return self._ucb(x, gp, self.kappa, self.prior_point_list)

    def update_params(self, success=None):
        # only change the learning rate chosen
        self._iters_counter += 1
        if self._kappa_decay < 1 and self._iters_counter > self._kappa_decay_delay:
            self.kappa *= self._kappa_decay

        if len(self.lr) > 0:
            self.lr[self.prior_pointer] *= self._lr_decay
            self.chosen_index = (self.chosen_index + 1) % len(self.prior_point_list)

    def compute_penalty_term(self, x, acq_score, prior):

        if len(self.lr) == 0:
            return 0

        chosen_index = self.chosen_index
        self.prior_pointer = chosen_index
        lr = self.lr[chosen_index]
        dist = np.linalg.norm(x - prior[chosen_index], axis=1)

        # Adding multiplier to make sure the penalty term is at the same scale as the UCB value
        if self.constant_multiplier is None:
            if isinstance(acq_score, float):
                self.constant_multiplier = acq_score / (lr * np.sqrt(dist) * 2)
            elif isinstance(acq_score, np.ndarray) or isinstance(acq_score, list):
                self.constant_multiplier = acq_score[0] / (lr * np.sqrt(dist[0]) * 2)

        penalty_term = lr * np.sqrt(dist) * self.constant_multiplier * 2
        return penalty_term

    def _ucb(self, x, gp, kappa, prior):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        ucb_score = mean + kappa * std
        penalty_term = self.compute_penalty_term(x, ucb_score, prior)
        return ucb_score - penalty_term

    def _ei(self, x, gp, y_max, xi, prior):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)
        a = (mean - y_max - xi)
        z = a / std
        ei_score = a * norm.cdf(z) + std * norm.pdf(z)
        penalty_term = self.compute_penalty_term(x, ei_score, prior)
        return ei_score - penalty_term

    def _poi(self, x, gp, y_max, xi, prior):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)
        z = (mean - y_max - xi) / std
        poi_score = norm.cdf(z)
        penalty_term = self.compute_penalty_term(x, poi_score, prior)
        return poi_score - penalty_term

    def delete_prior(self):
        if len(self.lr) > 0:
            self.lr = np.delete(self.lr, self.prior_pointer, 0)
            del(self.prior_point_list[self.prior_pointer])
            self.prior_pointer -= 1
