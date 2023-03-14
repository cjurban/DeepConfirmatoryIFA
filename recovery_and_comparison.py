#!/usr/bin/env python
#
# Author: Christopher J. Urban
# Affil.: L. L. Thurstone Psychometric Laboratory in the Dept. of Psychology
#         and Neuroscience, UNC-Chapel Hill
# E-mail: cjurban@live.unc.edu
#
# Purpose: Conduct recovery and comparison simulations.
#
###############################################################################

import sys
from pathlib import Path
import torch
from deepirtools import (
    IWAVE,
    manual_seed,
)

from src.mhrm import MHRM
from src.utils import (
    save_obj,
    load_obj,
)

sample_sizes = [500, 2500, 12500, 62500]
n_reps = 100

if __name__ == "__main__":
    rep = int(sys.argv[1]) % n_reps
    ss_idx = (int(sys.argv[1]) // n_reps) % len(sample_sizes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

################################################################################
#
# Recovery analyses
#
################################################################################

iw_samples = [1, 10, 100]

if __name__ == "__main__":
    iw_idx = (int(sys.argv[1]) // (n_reps * len(sample_sizes)))
    
# Hyperparameters.
lr = 5e-3
batch_size = 128

data_dir = "data/recovery/sample_size_" + str(ss_idx) + "/"
Y, _ = load_obj(data_dir + "data_" + str(rep))

res_dir = (
    "results/recovery/iw_samples_" + str(iw_idx) +
    "/sample_size_" + str(ss_idx) + "/"
)
Path(res_dir + "model/").mkdir(parents = True, exist_ok = True)
Path(res_dir + "ll/").mkdir(parents = True, exist_ok = True)

manual_seed(int((rep + 1) * (ss_idx + 1) * (iw_idx + 1)))

latent_size = 5
Q = torch.block_diag(*[torch.ones([10, 1])] * latent_size)
model = IWAVE(
    model_type = "grm",
    device = device,
    learning_rate = lr,
    Q = Q,
    latent_size = latent_size,
    n_cats = [5] * Y.shape[1],
    correlated_factors = torch.arange(latent_size).tolist(),
)
model.fit(Y, batch_size = batch_size, iw_samples = iw_samples[iw_idx])
ll = model.log_likelihood(Y, mc_samples = 10, iw_samples = 10)

save_obj(model, res_dir + "model/model_" + str(rep))
save_obj(ll, res_dir + "ll/ll_" + str(rep))

################################################################################
#
# Comparison analyses
#
################################################################################

sample_sizes = [625, 1250, 2500, 5000]

if iw_idx == 0:
    data_dir = "data/comparison/sample_size_" + str(ss_idx) + "/"
    Y, _ = load_obj(data_dir + "data_" + str(rep))

    res_dir = "results/comparison/i-wave/sample_size_" + str(ss_idx) + "/"
    Path(res_dir + "model/").mkdir(parents = True, exist_ok = True)
    Path(res_dir + "ll/").mkdir(parents = True, exist_ok = True)

    manual_seed(int((rep + 1) * (ss_idx + 1) * (iw_idx + 1)))

    latent_size = 10
    Q = torch.block_diag(*[torch.ones([10, 1])] * latent_size)
    model = IWAVE(
        model_type = "grm",
        device = device,
        learning_rate = lr,
        Q = Q,
        latent_size = latent_size,
        n_cats = [5] * Y.shape[1],
        correlated_factors = torch.arange(latent_size).tolist(),
    )
    model.fit(Y, batch_size = batch_size, iw_samples = 10)
    ll = model.log_likelihood(Y, mc_samples = 10, iw_samples = 10)

    save_obj(model, res_dir + "model/model_" + str(rep))
    save_obj(ll, res_dir + "ll/ll_" + str(rep))
    
    res_dir = "results/comparison/mh-rm/sample_size_" + str(ss_idx) + "/"
    Path(res_dir + "model/").mkdir(parents = True, exist_ok = True)

    model_spec = """
        F1  = 1-10
        F2  = 11-20
        F3  = 21-30
        F4  = 31-40
        F5  = 41-50
        F6  = 51-60
        F7  = 61-70
        F8  = 71-80
        F9  = 81-90
        F10 = 91-100
        COV = F1*F2*F3*F4*F5*F6*F7*F8*F9*F10
    """
    model = MHRM(
        model_specification = model_spec,
        seed = int((rep + 1) * (ss_idx + 1) * (iw_idx + 1)),
    )
    model.fit(Y)

    save_obj(model, res_dir + "model/model_" + str(rep))