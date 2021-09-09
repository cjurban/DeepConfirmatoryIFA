#!/usr/bin/env python
#
# Author: Christopher J. Urban
# Code Author: Christopher J. Urban
# Affil.: L. L. Thurstone Psychometric Laboratory in the
#         Dept. of Psychology and Neuroscience, UNC-Chapel Hill
# E-mail: cjurban@live.unc.edu
#
# Purpose: Some useful functions.
#
###############################################################################

import pickle
import math
import torch
from torch import nn
import torch.distributions as dist
import numpy as np
from scipy.stats import logistic
from scipy.optimize import linear_sum_assignment
from functools import reduce

EPS = 1e-16

# Save an object.
# https://stackoverflow.com/questions/19201290/how-to-save-a-dictionary-to-a-file
def save_obj(obj, name):
    with open(name + ".pkl", "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

# Load an object.
# https://stackoverflow.com/questions/19201290/how-to-save-a-dictionary-to-a-file
def load_obj(name):
    with open(name + ".pkl", "rb") as f:
        return pickle.load(f)

# Perform linear annealing.
# https://github.com/YannDubs/disentangling-vae/blob/master/disvae/models/losses.py
def linear_annealing(init,
                     fin,
                     step,
                     annealing_steps):
    if annealing_steps == 0:
        return fin
    assert isinstance(annealing_steps, int)
    assert fin > init
    delta = fin - init
    annealed = min(init + delta * step / annealing_steps, fin)
    return annealed

# Apply Xavier initialization to a Tensor with a given sparsity pattern.
def init_sparse_xavier_uniform_(init_mat,
                                pattern_mat):
    fan_in = init_mat.size(1)
    fan_outs = pattern_mat.sum(dim = 0)
    a = torch.sqrt(6 / (fan_in + fan_outs))
    init_mat.data[:, :] = dist.Uniform(-a, a).sample([init_mat.size(0)]).mul_(pattern_mat)
    
    return init_mat

# Fast sampling from multiple categorical distributions.
# https://stackoverflow.com/questions/34187130/fast-random-weighted-selection-across-all-rows-of-a-stochastic-matrix/34190035#34190035
def multi_categorical_sample(prob_matrix,
                             items):
    s = prob_matrix.cumsum(axis = 0)
    s[-1, :] = 1.
    r = np.random.rand(prob_matrix.shape[1])
    k = (s < r).sum(axis = 0)
    return items[k]

# Convert Tensor of Cartesian coordinates to spherical coordinates.
def cart2spher(cart_mat):
    n = cart_mat.size(1)
    spher_mat = torch.zeros_like(cart_mat)
    cos_mat = cart_mat[:, 1:n].cos()
    sin_mat = cart_mat[:, 1:n].sin().cumprod(1)
    
    spher_mat[:, 0] = cart_mat[:, 0] * cos_mat[:, 0]
    spher_mat[:, 1:(n - 1)] = cart_mat[:, 0].unsqueeze(1) * sin_mat[:, 0:(n - 2)] * cos_mat[:, 1:(n - 1)]
    spher_mat[:, -1] = cart_mat[:, 0] * sin_mat[:, -1]
    
    return spher_mat

# Caclucate thresholds with equal area under the logistic distribution.
def logistic_thresholds(n_cats):
    thresholds = [logistic.ppf((cat + 1)/ n_cats) for cat in range(n_cats - 1)]
    return np.asarray(thresholds, dtype = np.float32)
        
# Convert covariance matrix to correlation matrix.
# http://www.statsmodels.org/0.6.1/_modules/statsmodels/stats/moment_helpers.html
def cov2corr(cov,
             return_std = False):
    cov = np.asanyarray(cov)
    std_ = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std_, std_)

    return corr
        
# Change factor loadings signs if sum of loadings is negative.
def invert_factors(mat):
    mat = mat.copy()
    for col_idx in range(0, mat.shape[1]): 
        if np.sum(mat[:, col_idx]) < 0: 
            mat[:, col_idx] = -mat[:, col_idx]
            
    return mat

# Change correlation signs if sum of loadings is negative.
def invert_cor(cor,
               mat):
    cor = cor.copy()
    for col_idx in range(0, mat.shape[1]):
        if np.sum(mat[:, col_idx]) < 0:
            # Invert column and row.
            inv_col_idxs = np.delete(np.arange(cor.shape[1]), col_idx, 0)
            cor[:, inv_col_idxs] = -cor[:, inv_col_idxs]
            cor[inv_col_idxs, :] = -cor[inv_col_idxs, :]
            
    return cor