#!/usr/bin/env python
#
# Author: Christopher J. Urban
# Affil.: L. L. Thurstone Psychometric Laboratory in the Dept. of Psychology
#         and Neuroscience, UNC-Chapel Hill
# E-mail: cjurban@live.unc.edu
#
# Purpose: Simulate data from graded response model.
#
###############################################################################

import sys
import os
from pathlib import Path
import torch
from deepirtools import manual_seed

from src.utils import (
    save_obj,
    load_obj,
    GradedResponseModelSimulator,
)

################################################################################
#
# Best-fitting models
#
################################################################################

res_dir = "results/ipip-ffm/"

lls_ff = [load_obj(res_dir + "five-factor/ll/ll_" + str(i)) for i in range(10)]
models_ff = [load_obj(res_dir + "five-factor/model_prop/model_prop_" + str(i)) for i in range(10)]
best_idx_ff = lls_ff.index(max(lls_ff))
best_model_ff = models_ff[best_idx_ff]

lls_sf = [load_obj(res_dir + "seven-factor/ll/ll_" + str(i)) for i in range(10)]
models_sf = [load_obj(res_dir + "seven-factor/model_prop/model_prop_" + str(i)) for i in range(10)]
best_idx_sf = lls_sf.index(max(lls_sf))
best_model_sf = models_sf[best_idx_sf]

################################################################################
#
# Recovery simulations
#
################################################################################

sample_sizes = [500, 2500, 12500, 62500]
n_reps = 100

if __name__ == "__main__":
    ss_idx = int(sys.argv[1]) // 100
    rep = int(sys.argv[1]) % 100
    
if ss_idx <= 3:
    manual_seed((ss_idx + 1) * rep)

    data_dir = "data/recovery/sample_size_" + str(ss_idx) + "/"
    Path(data_dir).mkdir(parents = True, exist_ok = True)

    grm = GradedResponseModelSimulator(
        loadings = best_model_ff.loadings,
        intercepts = best_model_ff.intercepts,
        cov_mat = best_model_ff.cov,
        mean = best_model_ff.mean.unsqueeze(0),
    )
    data = grm.sample(
        sample_size = sample_sizes[ss_idx],
        return_scores = True,
    )
    save_obj(data, data_dir + "data_" + str(rep))

################################################################################
#
# Comparison simulations
#
################################################################################

sample_sizes = [1000, 2000, 4000, 8000]

if ss_idx <= 3:    
    manual_seed((ss_idx + 1) * rep)

    data_dir = "data/comparison/sample_size_" + str(ss_idx) + "/"
    Path(data_dir).mkdir(parents = True, exist_ok = True)

    grm = GradedResponseModelSimulator(
        loadings = torch.block_diag(
            *(2 * [best_model_ff.loadings]),
        ),
        intercepts = torch.cat(
            2 * [best_model_ff.intercepts], dim = 0,
        ),
        cov_mat = torch.block_diag(
            *(2 * [best_model_ff.cov]),
        ),
        mean = torch.cat(
            2 * [best_model_ff.mean], dim = 0,
        ).unsqueeze(0),
    )
    data = grm.sample(
        sample_size = sample_sizes[ss_idx],
        return_scores = True,
    )
    save_obj(data, data_dir + "data_" + str(rep))
    
################################################################################
#
# I-WAVE C2ST simulations
#
################################################################################
    
sample_sizes = [625, 1250, 2500, 5000, 10000]

manual_seed((ss_idx + 1) * rep)

data_dir = "data/c2st/five-factor/sample_size_" + str(ss_idx) + "/"
Path(data_dir).mkdir(parents = True, exist_ok = True)

grm = GradedResponseModelSimulator(
    loadings = best_model_ff.loadings,
    intercepts = best_model_ff.intercepts,
    cov_mat = best_model_ff.cov,
    mean = best_model_ff.mean.unsqueeze(0),
)
data = grm.sample(
    sample_size = sample_sizes[ss_idx],
    return_scores = True,
)
save_obj(data, data_dir + "data_" + str(rep))

data_dir = "data/c2st/seven-factor/sample_size_" + str(ss_idx) + "/"
Path(data_dir).mkdir(parents = True, exist_ok = True)

grm = GradedResponseModelSimulator(
    loadings = best_model_sf.loadings,
    intercepts = best_model_sf.intercepts,
    cov_mat = best_model_sf.cov,
    mean = best_model_sf.mean.unsqueeze(0),
)
data = grm.sample(
    sample_size = sample_sizes[ss_idx],
    return_scores = True,
)
save_obj(data, data_dir + "data_" + str(rep))