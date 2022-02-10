from bayes_opt.bayesian_optimization import BayesianOptimization
import pandas as pd
import numpy as np
from bayes_opt.event import Events
from .acq import ModifiedAcq


class ModifiedBayesianOptimization(BayesianOptimization):

    def __init__(self, f, pbounds, prior_point_list, random_state=None, verbose=2,
                 bounds_transformer=None):
        super(ModifiedBayesianOptimization, self).__init__(f, pbounds, random_state, verbose, bounds_transformer)
        self.prior_point_list = prior_point_list

        self._max_value = 0  # Record max value
        df_columns = list(pbounds.keys()) + ['target']
        self._df_max = pd.DataFrame([], columns=df_columns)  # Record all optimal points as experiment results
        self._df = pd.DataFrame([], columns=df_columns)  # Record all points as experiment results

        self.iter_results = []

    def probe(self, params, lazy=True):
        """Probe target of x"""
        if lazy:
            self._queue.add(params)
        else:
            target = self._space.probe(params)
            self.dispatch(Events.OPTIMIZATION_STEP)
            return target

    def maximize(self,
                 init_points=5,
                 n_iter=25,
                 acq='ucb',
                 kappa=2.576,
                 kappa_decay=1,
                 kappa_decay_delay=0,
                 xi=0.0,
                 **gp_params):
        """Mazimize your function"""
        self._prime_subscriptions()
        self.dispatch(Events.OPTIMIZATION_START)
        self._prime_queue(init_points)
        self.set_gp_params(**gp_params)

        util = ModifiedAcq(kind=acq,
                           kappa=kappa,
                           xi=xi,
                           kappa_decay=kappa_decay,
                           kappa_decay_delay=kappa_decay_delay,
                           prior_point_list=self.prior_point_list)
        iteration = 0
        while not self._queue.empty or iteration < n_iter:
            try:
                x_probe = next(self._queue)
            except StopIteration:
                util.update_params()
                x_probe = self.suggest(util)
                iteration += 1

            target = self.probe(x_probe, lazy=False)
            if isinstance(x_probe, dict):
                x_probe_dict = x_probe
            else:
                x_probe_dict = self._space.array_to_params(x_probe)
            x_probe_dict['target'] = target
            self._df = self._df.append(x_probe_dict, ignore_index=True)
            if target > self._max_value:
                self._df_max = self._df_max.append(x_probe_dict, ignore_index=True)
                self._max_value = target

            if self._bounds_transformer:
                self.set_bounds(
                    self._bounds_transformer.transform(self._space))

            if iteration < init_points:
                self.iter_results.append(target)
            if iteration >= init_points:
                if target < np.mean(self.iter_results):
                    # del(self.prior_point_list[util.prior_pointer])
                    util.delete_prior()

        self.dispatch(Events.OPTIMIZATION_END)
