#!/usr/bin/env python
#
# Author: Christopher J. Urban
# Code Author: Christopher J. Urban
# Affil.: L. L. Thurstone Psychometric Laboratory in the
#         Dept. of Psychology and Neuroscience, UNC-Chapel Hill
# E-mail: cjurban@live.unc.edu
#
# Purpose: Functions for conducting simulations.
#
###############################################################################

import torch
import torch.nn.functional as F
import torch.distributions as dist
import numpy as np
from scipy.linalg import block_diag
from sklearn.preprocessing import OneHotEncoder
from code.python.utils import *

# Dummy code a vector.
def dummy_code_vec(vec,
                   max_val):
    """
    Args:
        vec     (Tensor): Vector with ordinal entries.
        max_val (int): Maximum possible value for ordinal entries.
    """
    dummy_vec = torch.FloatTensor(vec.size(0), max_val)
    dummy_vec.zero_()
    
    return dummy_vec.scatter_(1, vec, 1)

# Simulate MIRT data.
def sim_mirt(n_obs,
             distribution,
             loadings,
             intercepts,
             n_cats,
             efficient = True,
             dummy_code = True):
    """
    Args:
        n_obs        (int): Number of observations to simulate.
        distribution (Distribution): Latent distribution object.
        loadings     (Tensor/array): Factor loadings matrix.
        intercepts   (Tensor/array): Vector of intercepts.
        n_cats       (list of int): List containing number of categories for each observed variable.
        efficient    (Boolean): Whether or not to sample efficiently. Inefficient sampling kept to
                                ensure experiments are reproducible.
    """
    # Define block diagonal loadings matrix.
    ones = [np.ones((n_cat - 1, 1)) for n_cat in n_cats]
    D = torch.from_numpy(block_diag(*ones)).float()
    loadings = torch.mm(D, loadings)
    
    # Sample factor scores.
    scores = distribution.sample(torch.Size([n_obs]))
    
    # Compute cumulative probailities.
    activations = F.linear(scores, loadings, intercepts)
    cum_probs = activations.sigmoid()

    # Compute item response probabilities.
    one_idxs = np.cumsum(n_cats) - 1
    zero_idxs = one_idxs - (np.asarray(n_cats) - 1)
    upper_probs = torch.ones(cum_probs.size(0), cum_probs.size(1) + (len(n_cats)))
    lower_probs = torch.zeros(cum_probs.size(0), cum_probs.size(1) + (len(n_cats)))
    upper_probs[:, torch.from_numpy(np.delete(np.arange(0, upper_probs.size(1), 1), one_idxs))] = cum_probs
    lower_probs[:, torch.from_numpy(np.delete(np.arange(0, lower_probs.size(1), 1), zero_idxs))] = cum_probs
    probs = (upper_probs - lower_probs).clamp(min = 1e-16)
    
    # Simulate data.
    idxs = np.concatenate((np.zeros(1), np.cumsum(n_cats)))
    ranges = [torch.from_numpy(np.arange(int(l), int(u))) for l, u in zip(idxs, idxs[1:])]
    
    if efficient:
        max_rng = max([rng.shape[0] for rng in ranges])
        probs_reshape = torch.cat([probs[:, rng] if rng.shape[0] == max_rng else
                                   torch.cat([probs[:, rng], torch.zeros([probs.shape[0], max_rng - rng.shape[0]])], dim = 1) for
                                   rng in ranges], dim = 0)
        cat_sample = multi_categorical_sample(probs_reshape.numpy().T, np.arange(max_rng)).reshape(len(n_cats), n_obs).T
        if dummy_code:
            enc = OneHotEncoder(categories = [np.arange(cat) for cat in n_cats])
            enc.fit(cat_sample)
            data = torch.from_numpy(enc.transform(cat_sample).toarray())
        else:
            data = torch.from_numpy(cat_sample)
    else:
        # Originally used this for sampling, which was very slow.
        if dummy_code:
            data = torch.cat([dummy_code_vec(torch.multinomial(probs[:, rng], 1), n_cats[i]) for
                              i, rng in enumerate(ranges)], dim = 1)
        else:
            data = torch.cat([torch.multinomial(probs[:, rng], 1) for rng in ranges], dim = 1)
    
    return data, scores

# Make data generating parameters for simulations.
def make_gen_params(orig_loadings,
                    orig_intercepts,
                    orig_n_cats,
                    new_n_cats,
                    orig_cov = None,
                    factor_mul = 1,
                    item_mul = 1):
    # Make generating loadings matrix.
    gen_loadings = torch.from_numpy(block_diag(*[orig_loadings.copy().repeat(item_mul, axis = 0) for _ in range(factor_mul)])).float()
    
    # Make generating intercepts.
    gen_intercepts = np.hstack([orig_intercepts.copy() for _ in range(item_mul * factor_mul)])
    n_items = item_mul * factor_mul * orig_loadings.shape[0]
    idxs = np.cumsum([n_cat - 1 for n_cat in ([1] + [orig_n_cats] * n_items)])
    sliced_ints = [gen_intercepts[idxs[i]:idxs[i + 1]] for i in range(len(idxs) - 1)]
    gen_intercepts = torch.Tensor(np.hstack([np.sort(np.random.choice(a, 
                                                                  size = new_n_cats - 1,
                                                                  replace = False),
                                                     axis = None) for a in sliced_ints])).float()

    # Make generating factor covariance matrix.
    if orig_cov is not None:
        gen_cov = torch.from_numpy(block_diag(*[orig_cov.copy() for _ in range(factor_mul)])).float()
    else:
        gen_cov = torch.eye(loadings.shape[1])
    
    return gen_loadings, gen_intercepts, gen_cov

# Simulate data from a baseline model.
def sim_base(data,
             n_cats,
             efficient = True,
             dummy_code = True):
    n_obs = data.shape[0]
    probs = torch.cat(n_obs * [data.sum(dim = 0, keepdim = True) / float(data.shape[0])], dim = 0)
    idxs = np.concatenate((np.zeros(1), np.cumsum(n_cats)))
    ranges = [torch.from_numpy(np.arange(int(l), int(u))) for l, u in zip(idxs, idxs[1:])]
    
    if efficient:
        max_rng = max([rng.shape[0] for rng in ranges])
        probs_reshape = torch.cat([probs[:, rng] if rng.shape[0] == max_rng else
                                   torch.cat([probs[:, rng], torch.zeros([probs.shape[0], max_rng - rng.shape[0]])], dim = 1) for
                                   rng in ranges], dim = 0)
        cat_sample = multi_categorical_sample(probs_reshape.numpy().T, np.arange(max_rng)).reshape(len(n_cats), n_obs).T
        if dummy_code:
            enc = OneHotEncoder(categories = [np.arange(cat) for cat in n_cats])
            enc.fit(cat_sample)
            data = torch.from_numpy(enc.transform(cat_sample).toarray())
        else:
            data = torch.from_numpy(cat_sample)
    else:
        # Originally used this for sampling, which was very slow.
        if dummy_code:
            data = torch.cat([dummy_code_vec(torch.multinomial(probs[:, rng], 1), n_cats[i]) for
                              i, rng in enumerate(ranges)], dim = 1)
        else:
            data = torch.cat([torch.multinomial(probs[:, rng], 1) for rng in ranges], dim = 1)
    return data