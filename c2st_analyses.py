#!/usr/bin/env python
#
# Author: Christopher J. Urban
# Affil.: L. L. Thurstone Psychometric Laboratory in the Dept. of Psychology
#         and Neuroscience, UNC-Chapel Hill
# E-mail: cjurban@live.unc.edu
#
# Purpose: Conduct C2ST simulations.
#
###############################################################################

import sys
from pathlib import Path
import torch
from torch.distributions import Uniform
import math
from deepirtools import (
    IWAVE,
    manual_seed,
)

from src.utils import (
    save_obj,
    load_obj,
)
from src.c2st import (
    C2ST,
    DistributionModel,
    MultinomialBaselineModel,
    c2st_rfi,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

################################################################################
#
# TI error and power simulations
#
################################################################################

rr_types = ["t1_error", "power"]
sample_sizes = [250, 500, 1000, 2500, 5000, 10000]
n_reps = 100

if __name__ == "__main__":
    rep = int(sys.argv[1]) % n_reps
    ss_idx = (int(sys.argv[1]) // n_reps) % len(sample_sizes)
    rr_idx = (int(sys.argv[1]) // (n_reps * len(sample_sizes)))
    
rr_type = rr_types[rr_idx]
sample_size = sample_sizes[ss_idx]

# Hyperparameters.
lr = 5e-3
batch_size = 128
max_iter = 100000

res_dir = (
    "results/c2st/" + rr_type + "/sample_size_" + str(ss_idx) + "/"
)
Path(res_dir).mkdir(parents = True, exist_ok = True)

data = Uniform(torch.tensor([0.]), torch.tensor([1.])).sample([sample_size])
if rr_type == "t1_error":
    model = DistributionModel(Uniform(torch.tensor([0.05]), torch.tensor([1.05])))
elif rr_type == "power":
    model = DistributionModel(Uniform(torch.tensor([0.1]), torch.tensor([1.1])))
    
c2st = C2ST(
    input_size = 1,
    device = device,
    learning_rate = lr,
)
c2st.global_fit_test(
    model = model,
    data_real = data,
    delta = 0.025,
    batch_size = batch_size,
    max_epochs = math.floor(max_iter * batch_size / sample_size),
)

save_obj(c2st, res_dir + "c2st_" + str(rep))

################################################################################
#
# I-WAVE C2ST simulations
#
################################################################################

sample_sizes = [625, 1250, 2500, 5000, 10000]

if rr_idx == 0 and ss_idx < len(sample_sizes):
    
    ################################################################################
    #
    # Five-factor data, five-factor fitted
    #
    ################################################################################

    if __name__ == "__main__":
        ss_idx = int(sys.argv[1]) // n_reps

    data_dir = "data/c2st/five-factor/sample_size_" + str(ss_idx) + "/"
    Y, _ = load_obj(data_dir + "data_" + str(rep))
    sample_size, n_items = Y.shape

    res_dir = (
        "results/c2st/gen_five-factor/fit_five-factor/" +
        "sample_size_" + str(ss_idx) + "/"
    )
    Path(res_dir + "model_prop/").mkdir(parents = True, exist_ok = True)
    Path(res_dir + "ll/").mkdir(parents = True, exist_ok = True)
    Path(res_dir + "c2st_prop/").mkdir(parents = True, exist_ok = True)
    Path(res_dir + "c2st_base/").mkdir(parents = True, exist_ok = True)
    Path(res_dir + "c2st_rfi/").mkdir(parents = True, exist_ok = True)

    manual_seed((ss_idx + 1) * rep)

    # Constraints.
    latent_size = 5
    Q = torch.block_diag(*[torch.ones([10, 1])] * latent_size)
    A = torch.block_diag(
        *(latent_size * [torch.eye(10), torch.zeros([50, 50])] +
          [torch.zeros([50, 50])])
    )
    A[266, 266] += 1; A[267, 266] += 1; A[340, 340] += 1; A[347, 340] += 1

    # Model fitting.
    model_prop = IWAVE(
        model_type = "grm",
        device = device,
        learning_rate = lr,
        Q = Q,
        latent_size = latent_size,
        n_cats = [5] * n_items,
        correlated_factors = torch.arange(latent_size).tolist(),
    )
    model_prop.fit(Y, batch_size = batch_size, iw_samples = 10)
    ll = model_prop.log_likelihood(Y, mc_samples = 10, iw_samples = 10)

    save_obj(model_prop, res_dir + "model_prop/model_prop_" + str(rep))
    save_obj(ll, res_dir + "ll/ll_" + str(rep))

    # Goodness-of-fit.
    c2st_prop = C2ST(
        input_size = n_items,
        device = device,
        learning_rate = lr,
    )
    c2st_prop.global_fit_test(
        model = model_prop,
        data_real = Y,
        delta = 0.05,
        batch_size = batch_size,
        max_epochs = math.floor(max_iter * batch_size / sample_size),
    )
    c2st_prop.permutation_importance()

    model_base = MultinomialBaselineModel(Y)
    c2st_base = C2ST(
        input_size = n_items,
        device = device,
        learning_rate = lr,
    )
    c2st_base.global_fit_test(
        model = model_base,
        data_real = Y,
        delta = 0.05,
        batch_size = batch_size,
        max_epochs = math.floor(max_iter * batch_size / sample_size),
    )
    c2st_base.permutation_importance()

    rfi = c2st_rfi(
        model_prop = model_prop,
        model_base = model_base,
        c2st_prop = c2st_prop,
        c2st_base = c2st_base,
    )

    save_obj(c2st_prop, res_dir + "c2st_prop/c2st_prop_" + str(rep))
    save_obj(c2st_base, res_dir + "c2st_base/c2st_base_" + str(rep))
    save_obj(rfi, res_dir + "c2st_rfi/c2st_rfi_" + str(rep))

    ################################################################################
    #
    # Five-factor data, seven-factor fitted
    #
    ################################################################################

    res_dir = (
        "results/c2st/gen_five-factor/fit_seven-factor/" +
        "sample_size_" + str(ss_idx) + "/"
    )
    Path(res_dir + "model_prop/").mkdir(parents = True, exist_ok = True)
    Path(res_dir + "ll/").mkdir(parents = True, exist_ok = True)
    Path(res_dir + "c2st_prop/").mkdir(parents = True, exist_ok = True)
    Path(res_dir + "c2st_base/").mkdir(parents = True, exist_ok = True)
    Path(res_dir + "c2st_rfi/").mkdir(parents = True, exist_ok = True)

    # Model fitting.
    model_prop = IWAVE(
        model_type = "grm",
        device = device,
        learning_rate = lr,
        A = A,
        latent_size = latent_size + 2,
        n_cats = [5] * n_items,
        correlated_factors = torch.arange(latent_size).tolist(),
    )
    model_prop.fit(Y, batch_size = batch_size, iw_samples = 10)
    ll = model_prop.log_likelihood(Y, mc_samples = 10, iw_samples = 10)

    save_obj(model_prop, res_dir + "model_prop/model_prop_" + str(rep))
    save_obj(ll, res_dir + "ll/ll_" + str(rep))

    # Goodness-of-fit.
    c2st_prop = C2ST(
        input_size = n_items,
        device = device,
        learning_rate = lr,
    )
    c2st_prop.global_fit_test(
        model = model_prop,
        data_real = Y,
        delta = 0.05,
        batch_size = batch_size,
        max_epochs = math.floor(max_iter * batch_size / sample_size),
    )
    c2st_prop.permutation_importance()

    model_base = MultinomialBaselineModel(Y)
    c2st_base = C2ST(
        input_size = n_items,
        device = device,
        learning_rate = lr,
    )
    c2st_base.global_fit_test(
        model = model_base,
        data_real = Y,
        delta = 0.05,
        batch_size = batch_size,
        max_epochs = math.floor(max_iter * batch_size / sample_size),
    )
    c2st_base.permutation_importance()

    rfi = c2st_rfi(
        model_prop = model_prop,
        model_base = model_base,
        c2st_prop = c2st_prop,
        c2st_base = c2st_base,
    )

    save_obj(c2st_prop, res_dir + "c2st_prop/c2st_prop_" + str(rep))
    save_obj(c2st_base, res_dir + "c2st_base/c2st_base_" + str(rep))
    save_obj(rfi, res_dir + "c2st_rfi/c2st_rfi_" + str(rep))

    ################################################################################
    #
    # Seven-factor data, five-factor fitted
    #
    ################################################################################

    data_dir = "data/c2st/seven-factor/sample_size_" + str(ss_idx) + "/"
    Y, _ = load_obj(data_dir + "data_" + str(rep))

    res_dir = (
        "results/c2st/gen_seven-factor/fit_five-factor/" +
        "sample_size_" + str(ss_idx) + "/"
    )
    Path(res_dir + "model_prop/").mkdir(parents = True, exist_ok = True)
    Path(res_dir + "ll/").mkdir(parents = True, exist_ok = True)
    Path(res_dir + "c2st_prop/").mkdir(parents = True, exist_ok = True)
    Path(res_dir + "c2st_base/").mkdir(parents = True, exist_ok = True)
    Path(res_dir + "c2st_rfi/").mkdir(parents = True, exist_ok = True)

    # Model fitting.
    model_prop = IWAVE(
        model_type = "grm",
        device = device,
        learning_rate = lr,
        Q = Q,
        latent_size = latent_size,
        n_cats = [5] * n_items,
        correlated_factors = torch.arange(latent_size).tolist(),
    )
    model_prop.fit(Y, batch_size = batch_size, iw_samples = 10)
    ll = model_prop.log_likelihood(Y, mc_samples = 10, iw_samples = 10)

    save_obj(model_prop, res_dir + "model_prop/model_prop_" + str(rep))
    save_obj(ll, res_dir + "ll/ll_" + str(rep))

    # Goodness-of-fit.
    c2st_prop = C2ST(
        input_size = n_items,
        device = device,
        learning_rate = lr,
    )
    c2st_prop.global_fit_test(
        model = model_prop,
        data_real = Y,
        delta = 0.05,
        batch_size = batch_size,
        max_epochs = math.floor(max_iter * batch_size / sample_size),
    )
    c2st_prop.permutation_importance()

    model_base = MultinomialBaselineModel(Y)
    c2st_base = C2ST(
        input_size = n_items,
        device = device,
        learning_rate = lr,
    )
    c2st_base.global_fit_test(
        model = model_base,
        data_real = Y,
        delta = 0.05,
        batch_size = batch_size,
        max_epochs = math.floor(max_iter * batch_size / sample_size),
    )
    c2st_base.permutation_importance()

    rfi = c2st_rfi(
        model_prop = model_prop,
        model_base = model_base,
        c2st_prop = c2st_prop,
        c2st_base = c2st_base,
    )

    save_obj(c2st_prop, res_dir + "c2st_prop/c2st_prop_" + str(rep))
    save_obj(c2st_base, res_dir + "c2st_base/c2st_base_" + str(rep))
    save_obj(rfi, res_dir + "c2st_rfi/c2st_rfi_" + str(rep))

    ################################################################################
    #
    # Seven-factor data, seven-factor fitted
    #
    ################################################################################

    res_dir = (
        "results/c2st/gen_seven-factor/fit_seven-factor/" +
        "sample_size_" + str(ss_idx) + "/"
    )
    Path(res_dir + "model_prop/").mkdir(parents = True, exist_ok = True)
    Path(res_dir + "ll/").mkdir(parents = True, exist_ok = True)
    Path(res_dir + "c2st_prop/").mkdir(parents = True, exist_ok = True)
    Path(res_dir + "c2st_base/").mkdir(parents = True, exist_ok = True)
    Path(res_dir + "c2st_rfi/").mkdir(parents = True, exist_ok = True)

    # Model fitting.
    model_prop = IWAVE(
        model_type = "grm",
        device = device,
        learning_rate = lr,
        A = A,
        latent_size = latent_size + 2,
        n_cats = [5] * n_items,
        correlated_factors = torch.arange(latent_size).tolist(),
    )
    model_prop.fit(Y, batch_size = batch_size, iw_samples = 10)
    ll = model_prop.log_likelihood(Y, mc_samples = 10, iw_samples = 10)

    save_obj(model_prop, res_dir + "model_prop/model_prop_" + str(rep))
    save_obj(ll, res_dir + "ll/ll_" + str(rep))

    # Goodness-of-fit.
    c2st_prop = C2ST(
        input_size = n_items,
        device = device,
        learning_rate = lr,
    )
    c2st_prop.global_fit_test(
        model = model_prop,
        data_real = Y,
        delta = 0.05,
        batch_size = batch_size,
        max_epochs = math.floor(max_iter * batch_size / sample_size),
    )
    c2st_prop.permutation_importance()

    model_base = MultinomialBaselineModel(Y)
    c2st_base = C2ST(
        input_size = n_items,
        device = device,
        learning_rate = lr,
    )
    c2st_base.global_fit_test(
        model = model_base,
        data_real = Y,
        delta = 0.05,
        batch_size = batch_size,
        max_epochs = math.floor(max_iter * batch_size / sample_size),
    )
    c2st_base.permutation_importance()

    rfi = c2st_rfi(
        model_prop = model_prop,
        model_base = model_base,
        c2st_prop = c2st_prop,
        c2st_base = c2st_base,
    )

    save_obj(c2st_prop, res_dir + "c2st_prop/c2st_prop_" + str(rep))
    save_obj(c2st_base, res_dir + "c2st_base/c2st_base_" + str(rep))
    save_obj(rfi, res_dir + "c2st_rfi/c2st_rfi_" + str(rep))