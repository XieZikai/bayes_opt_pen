This package contains penalised acquisition functions in file acq.py.

To use these functions:

1. Manually adding prior knowledge into the Gaussian process regressor:

gp.prior_point_list = prior_point_list

prior_point_list should be a list of tensors(data points of priors).

2. Simply change the acquisition function to penalised acquisition function and bind it into
the optimization process.