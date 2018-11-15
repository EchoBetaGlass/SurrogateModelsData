"""Create synthetic datasets for surrogate training."""

# %% imports
from optproblems import dtlz
from pyDOE import lhs
import numpy as np
import pandas as pd
from scipy.stats.distributions import norm
import pickle as pk
from os import getcwd

# %%
problems = {'DTLZ1': dtlz.DTLZ1,
            'DTLZ2': dtlz.DTLZ2,
            'DTLZ3': dtlz.DTLZ3,
            'DTLZ4': dtlz.DTLZ4,
            'DTLZ5': dtlz.DTLZ5,
            'DTLZ6': dtlz.DTLZ6,
            'DTLZ7': dtlz.DTLZ7}
folder = getcwd() + '/dtlzdatasets/'
# %%
num_vars = [5, 10, 20]
num_obj = 2
num_samples = [100, 500, 1000]
distribution = ['uniform', 'normal']
noise = [False, True]
missing_data = [False, True]
# %% Parameters for normal distribution and noise
means = [0.4, 0.6]
stdvs = [0.25, 0.2]
noise_mean = 0
noise_std = 0.1
miss_fraction = 0.1
# %% Creation of datasets
for problem in problems:
    for num_var in num_vars:
        print(problems[problem])
        objective = problems[problem](num_obj, num_var)
        for num_sample in num_samples:
            var_names = ['x{0}'.format(x) for x in range(num_var)]
            obj_names = ['f1', 'f2']
            filename = folder + problem + '_' + str(num_var) + '_' + str(num_sample)
            var = lhs(num_var, num_sample)
            obj = [objective(x) for x in var]
            data_uniform = np.hstack((var, obj))
            data_noisy = data_uniform
            data_uniform = pd.DataFrame(data_uniform,
                                        columns=var_names+obj_names)
            pk.dump(data_uniform, open((filename + 'uniform.p'), 'wb'))
            for i in range(num_var+num_obj):
                data_noisy[:, i] = (data_noisy[:, i]
                                    + np.random.normal(noise_mean, 
                                                       noise_std, num_sample))
            data_noisy = pd.DataFrame(data_noisy,
                                      columns=var_names+obj_names)
            pk.dump(data_noisy, open((filename + 'uniform_noisy.p'), 'wb'))
            var_norm = var
            for i in range(2):
                var_norm[:, i] = norm(loc=means[i],
                                      scale=stdvs[i]).ppf(var[:, i])
            var_norm[var_norm > 1] = 1
            var_norm[var_norm < 0] = 0
            obj_norm = [objective(x) for x in var_norm]
            data_normalized = np.hstack((var, obj))
            data_noisy = data_normalized
            data_normalized = pd.DataFrame(data_normalized,
                                           columns=var_names+obj_names)
            pk.dump(data_normalized, open((filename + 'normal.p'), 'wb'))
            for i in range(num_var+num_obj):
                data_noisy[:, i] = (data_noisy[:, i]
                                    + np.random.normal(noise_mean, 
                                                       noise_std, num_sample))
            data_noisy = pd.DataFrame(data_noisy,
                                      columns=var_names+obj_names)
            pk.dump(data_noisy, open((filename + 'normal_noisy.p'), 'wb'))
