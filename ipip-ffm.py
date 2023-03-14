#!/usr/bin/env python
#
# Author: Christopher J. Urban
# Affil.: L. L. Thurstone Psychometric Laboratory in the Dept. of Psychology
#         and Neuroscience, UNC-Chapel Hill
# E-mail: cjurban@live.unc.edu
#
# Purpose: Conduct IPIP-FFM analyses.
#
###############################################################################

import sys
import os
import urllib.request
import shutil
from pathlib import Path
import pandas as pd
import torch
import numpy as np
import math
from deepirtools import (
    IWAVE,
    manual_seed,
)

from src.utils import save_obj
from src.c2st import (
    C2ST,
    MultinomialBaselineModel,
    c2st_rfi,
)

if __name__ == "__main__":
    rep = int(sys.argv[1])
manual_seed(rep)

# Download data.
filepath = "./data/IPIP-FFM-data-8Nov2018/data-final.csv"
if not os.path.exists(filepath):
    os.makedirs(os.path.dirname("./data/"), exist_ok = True)
    urllib.request.urlretrieve(
        "https://openpsychometrics.org/_rawdata/IPIP-FFM-data-8Nov2018.zip",
        "./data/ipip-ffm.zip"
    )
    shutil.unpack_archive("./data/ipip-ffm.zip", "./data/")

# Pre-processing.
df = pd.read_csv(filepath, sep = "\t", header = 0)
df = df[df["IPC"] == 1] # Drop multiple submissions from same IP address.
df.iloc[:, :100] = df.iloc[:, :100].dropna() # Drop people with all NaN responses.
df = df[df.iloc[:, 0:50].sum(1) > 0] # Drop people with all missing responses.
rc_items = [
    "EXT2", "EXT4", "EXT6", "EXT8", "EXT10", "AGR1", "AGR3", "AGR5", "AGR7",
    "CSN2", "CSN4", "CSN6", "CSN8", "EST2", "EST4", "OPN2", "OPN4", "OPN6",
]
df[rc_items] = ((1 - df[rc_items] / 5) * 5 + 1).mask(lambda col: col == 6, 0) # Reverse-code reverse-coded items.
Y = torch.from_numpy(df.iloc[:, :50].to_numpy()) - 1 # Collect item responses.
T = torch.from_numpy(df.iloc[:, 50:100].to_numpy()) / 1000 # Collect response times in seconds.
keeps = (
    (((T < 0).sum(dim = 1) == 0) & # Negative response times.
    ((T > 100).sum(dim = 1) == 0)) & # Long response times.
    (Y.eq(-1).sum(dim=1).eq(0)) # Missing item responses.
)
Y = Y[keeps]

sample_size, n_items = Y.shape
n_reps = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

################################################################################
#
# Five-factor model analyses
#
################################################################################

res_dir = "results/ipip-ffm/five-factor/"
Path(res_dir + "model_prop/").mkdir(parents = True, exist_ok = True)
Path(res_dir + "ll/").mkdir(parents = True, exist_ok = True)
Path(res_dir + "c2st_prop/").mkdir(parents = True, exist_ok = True)
Path(res_dir + "c2st_base/").mkdir(parents = True, exist_ok = True)
Path(res_dir + "c2st_rfi/").mkdir(parents = True, exist_ok = True)

# Constraints.
latent_size = 5
Q = torch.block_diag(*[torch.ones([10, 1])] * latent_size)

# Hyperparameters.
lr = 5e-3
batch_size = 128
max_iter = 100000

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
# Seven-factor model analyses
#
################################################################################

res_dir = "results/ipip-ffm/seven-factor/"
Path(res_dir + "model_prop/").mkdir(parents = True, exist_ok = True)
Path(res_dir + "ll/").mkdir(parents = True, exist_ok = True)
Path(res_dir + "c2st_prop/").mkdir(parents = True, exist_ok = True)
Path(res_dir + "c2st_base/").mkdir(parents = True, exist_ok = True)
Path(res_dir + "c2st_rfi/").mkdir(parents = True, exist_ok = True)

# Constraints.
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
    delta = 0.1,
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
    delta = 0.1,
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