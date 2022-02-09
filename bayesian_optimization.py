from bayes_opt.bayesian_optimization import BayesianOptimization as OriginalBO
from bayesian_optimization_with_penalty import ModifiedBayesianOptimization


def BayesianOptimization(f, pbounds, prior_point_list=None, random_state=None, verbose=2,
                         bounds_transformer=None,):
    """
    This is a wrapper of choosing BO algorithm.
    """
    if prior_point_list is None:
        return OriginalBO(f, pbounds, random_state, verbose, bounds_transformer)
    else:
        assert isinstance(prior_point_list, list), 'Prior point list must be a list!'
        for prior in prior_point_list:
            assert len(prior) == len(pbounds), 'Wrong prior point length! Prior: '.format(prior)
        return ModifiedBayesianOptimization(f, pbounds, prior_point_list, random_state, verbose, bounds_transformer)
