#!/usr/bin/env python
#
# Author: Christopher J. Urban
# Affil.: L. L. Thurstone Psychometric Laboratory in the Dept. of Psychology
#         and Neuroscience, UNC-Chapel Hill
# E-mail: cjurban@live.unc.edu
#
# Purpose: Create figures.
#
###############################################################################

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from itertools import product

from src.c2st import approx_power
from src.utils import (
    load_obj,
)
from src.fig_utils import (
    importances_plt,
    boxplots,
    time_plt,
    rr_acc_plt,
)

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 14

import sys
sys.modules["src.python"] = sys.modules["src"] # Patch due to directory change.

################################################################################
#
# IPIP-FFM figures
#
################################################################################

fig_dir = "figures/ipip-ffm/"
Path(fig_dir).mkdir(parents = True, exist_ok = True)

res_dir = "results/ipip-ffm/"

n_reps = 10

for model_type in ("five-factor", "seven-factor"):
    lls = [load_obj(res_dir + model_type + "/ll/ll_" + str(i)) for i in range(n_reps)]
    models = [load_obj(res_dir + model_type + "/model_prop/model_prop_" + str(i)) for i in range(n_reps)]
    c2st_props = [load_obj(res_dir + model_type + "/c2st_prop/c2st_prop_" + str(i)) for i in range(n_reps)]
    c2st_bases = [load_obj(res_dir + model_type + "/c2st_base/c2st_base_" + str(i)) for i in range(n_reps)]
    c2st_rfis = [load_obj(res_dir + model_type + "/c2st_rfi/c2st_rfi_" + str(i)) for i in range(n_reps)]

    best_idx = lls.index(max(lls))
    best_model = models.pop(best_idx)
    if model_type == "five-factor":
        best_model_ff = best_model
    elif model_type == "seven-factor":
        best_model_sf = best_model
    
    ldgs_rmses = torch.stack(
        [m.loadings.add(-best_model.loadings).pow(2) for m in models], dim = 0,
    ).mean(dim = 0).sqrt()[best_model.loadings.ne(0)]
    ints_rmses = torch.stack(
        [m.intercepts.add(-best_model.intercepts).pow(2) for m in models], dim = 0,
    ).mean(dim = 0).sqrt()
    cor_rmses = torch.stack(
        [m.cov.add(-best_model.cov).pow(2) for m in models], dim = 0,
    ).mean(dim = 0).sqrt()[best_model.cov.tril(diagonal = -1).ne(0)]
    run_times = [m.timerecords["fit"] for m in models]

    np.savetxt(
        fig_dir + "best_" + model_type + "_loadings.txt",
        best_model.loadings.numpy(), fmt = "%.2f",
    )
    np.savetxt(
        fig_dir + "best_" + model_type + "_intercepts.txt",
        best_model.intercepts.numpy(), fmt = "%.2f",
    )
    np.savetxt(
        fig_dir + "best_" + model_type + "_correlations.txt",
        best_model.cov.numpy(), fmt = "%.2f",
    )
    np.savetxt(
        fig_dir + model_type + "_loadings_rmses.txt",
        np.array((ldgs_rmses.mean().numpy(), ldgs_rmses.std().numpy())), fmt = "%.2f",
    )
    np.savetxt(
        fig_dir + model_type + "_intercepts_rmses.txt",
        np.array((ints_rmses.mean().numpy(), ints_rmses.std().numpy())), fmt = "%.2f",
    )
    np.savetxt(
        fig_dir + model_type + "_correlations_rmses.txt",
        np.array((cor_rmses.mean().numpy(), cor_rmses.std().numpy())), fmt = "%.2f",
    )
    np.savetxt(
        fig_dir + model_type + "_run_times.txt",
        np.array((np.mean(run_times), np.std(run_times, ddof = 1))), fmt = "%.2f",
    )
    
    acc_props = [c.acc for c in c2st_props]
    p_val_props = [c.p_val for c in c2st_props]
    sample_props = [c.timerecords["sample"] for c in c2st_props]
    fit_props = [c.timerecords["fit"] for c in c2st_props]
    importances_props = [c.timerecords["importances"] for c in c2st_props]
    acc_bases = [c.acc for c in c2st_bases]
    p_val_bases = [c.p_val for c in c2st_bases]
    sample_bases = [c.timerecords["sample"] for c in c2st_bases]
    fit_bases = [c.timerecords["fit"] for c in c2st_bases]
    
    np.savetxt(
        fig_dir + model_type + "_c2st_res.txt",
        np.array(
            (
                (np.mean(sample_bases), np.std(sample_bases, ddof = 0),
                 np.mean(sample_props), np.std(sample_props, ddof = 0),),
                (np.mean(fit_bases), np.std(fit_bases, ddof = 0),
                 np.mean(fit_props), np.std(fit_props, ddof = 0),),
                (np.mean(acc_bases), np.std(acc_bases, ddof = 0),
                 np.mean(acc_props), np.std(acc_props, ddof = 0),),
                (np.nan, np.nan,
                 np.mean(c2st_rfis), np.std(c2st_rfis, ddof = 0),),
            )
        ), fmt = "%.2f",
    )
    
    fig = importances_plt(
        c2sts = c2st_props,
        x_labels = ["EXT\n(Items 1–10)",
                    "EST\n(Items 11–20)",
                    "AGR\n(Items 21–30)",
                    "CSN\n(Items 31–40)",
                    "OPN\n(Items 41–50)"],
        ylim = [-0.0025, 0.0525],
        hatches = [16, 17, 19, 40, 47],
    )
    pdf = matplotlib.backends.backend_pdf.PdfPages(
        fig_dir + "ipip-ffm_" + model_type + "_importances_plt.pdf"
    )
    pdf.savefig(fig, dpi = 300)
    pdf.close()
    
################################################################################
#
# Recovery figures
#
################################################################################

fig_dir = "figures/recovery/"
Path(fig_dir).mkdir(parents = True, exist_ok = True)

res_dir = "results/recovery/"

sample_sizes = [500, 2500, 12500, 62500]
iw_samples = [1, 10, 100]
cond_labels = ["1 IW Sample", "10 IW Samples", "100 IW Samples"]
n_reps = 100

models = [[[
    load_obj(
        res_dir + "iw_samples_" + str(i) +
                  "/sample_size_" + str(j) +
                  "/model/model_" + str(k)
            ) for k in range(n_reps)
        ] for j in range(len(sample_sizes))
    ] for i in range(len(iw_samples))
]

models[2][0].pop(94) # Divergent replication.

ldgs_deltas = [[[
    m3.loadings.add(-best_model_ff.loadings)[best_model_ff.loadings.ne(0)]
            for m3 in m2
        ] for m2 in m1
    ] for m1 in models
]
ints_deltas = [[[
    m3.intercepts.add(-best_model_ff.intercepts).reshape(-1)
            for m3 in m2
        ] for m2 in m1
    ] for m1 in models
]
cor_deltas = [[[
    m3.cov.add(-best_model_ff.cov)[best_model_ff.cov.tril(diagonal = -1).ne(0)]
            for m3 in m2
        ] for m2 in m1
    ] for m1 in models
]

ldgs_biases = [[
    torch.stack(d2, dim = 0).mean(dim = 0)
        for d2 in d1
    ] for d1 in ldgs_deltas
]
ldgs_mses = [[
    torch.stack(d2, dim = 0).pow(2).mean(dim = 0)
        for d2 in d1
    ] for d1 in ldgs_deltas
]
ldgs_bias_fig = boxplots(
    vals = ldgs_biases,
    sample_sizes = sample_sizes,
    y_label = "Bias",
    cond_labels = cond_labels,
    hline_locs = [0.],
)
ldgs_mse_fig = boxplots(
    vals = ldgs_mses,
    sample_sizes = sample_sizes,
    y_label = "MSE",
    cond_labels = cond_labels,
    ylim = [1e-4, 1e-1],
    log_scale = True,
)

ints_biases = [[
    torch.stack(d2, dim = 0).mean(dim = 0)
        for d2 in d1
    ] for d1 in ints_deltas
]
ints_mses = [[
    torch.stack(d2, dim = 0).pow(2).mean(dim = 0)
        for d2 in d1
    ] for d1 in ints_deltas
]
ints_bias_fig = boxplots(
    vals = ints_biases,
    sample_sizes = sample_sizes,
    y_label = "Bias",
    cond_labels = cond_labels,
    hline_locs = [0.],
)
ints_mse_fig = boxplots(
    vals = ints_mses,
    sample_sizes = sample_sizes,
    y_label = "MSE",
    cond_labels = cond_labels,
    ylim = [1e-4, 1],
    log_scale = True,
)

cor_biases = [[
    torch.stack(d2, dim = 0).mean(dim = 0)
        for d2 in d1
    ] for d1 in cor_deltas
]
cor_mses = [[
    torch.stack(d2, dim = 0).pow(2).mean(dim = 0)
        for d2 in d1
    ] for d1 in cor_deltas
]
cor_bias_fig = boxplots(
    vals = cor_biases,
    sample_sizes = sample_sizes,
    y_label = "Bias",
    cond_labels = cond_labels,
    hline_locs = [0.],
)
cor_mse_fig = boxplots(
    vals = cor_mses,
    sample_sizes = sample_sizes,
    y_label = "MSE",
    cond_labels = cond_labels,
    ylim = [1e-5, 1e-2],
    log_scale = True,
)

for lab, fig in zip(
    ("ldgs", "ints", "cor"),
    (ldgs_bias_fig, ints_bias_fig, cor_bias_fig),
):
    pdf = matplotlib.backends.backend_pdf.PdfPages(
        fig_dir + "recovery_" + lab + "_bias_plt.pdf"
    )
    pdf.savefig(fig, dpi = 300)
    pdf.close()
    
for lab, fig in zip(
    ("ldgs", "ints", "cor"),
    (ldgs_mse_fig, ints_mse_fig, cor_mse_fig),
):
    pdf = matplotlib.backends.backend_pdf.PdfPages(
        fig_dir + "recovery_" + lab + "_mse_plt.pdf")
    pdf.savefig(fig, dpi = 300)
    pdf.close()

time_fig = time_plt(
    models = models,
    sample_sizes = sample_sizes,
    ylim = [0, 10**3],
    cond_labels = cond_labels,
    legend_loc = "center right",
    x_trans = (-0.05, 0.05, 0),
)
pdf = matplotlib.backends.backend_pdf.PdfPages(
    fig_dir + "recovery_time_plt.pdf"
)
pdf.savefig(time_fig, dpi = 300)
pdf.close()

################################################################################
#
# Comparison figures
#
################################################################################

fig_dir = "figures/comparison/"
Path(fig_dir).mkdir(parents = True, exist_ok = True)

res_dir = "results/comparison/"

sample_sizes = [625, 1250, 2500, 5000]
estimators = ["i-wave", "mh-rm"]
cond_labels = ["I-WAVE", "MH-RM"]
n_reps = 100

models = [[[
    load_obj(
        res_dir + e +
                  "/sample_size_" + str(j) +
                  "/model/model_" + str(k)
            ) for k in range(n_reps)
        ] for j in range(len(sample_sizes))
    ] for e in estimators
]

for i in range(len(sample_sizes)):
    for j in range(n_reps):
        models[1][i][j].timerecords["fit"] = (
            models[1][i][j].timerecords["MH_draws"] +
            models[1][i][j].timerecords["Mstep"]
        )

loadings = torch.block_diag(
    *(2 * [best_model_ff.loadings]),
)
intercepts = torch.cat(
    2 * [best_model_ff.intercepts], dim = 0,
)
cor_mat = torch.block_diag(
    *(2 * [best_model_ff.cov]),
)

ldgs_deltas = [[[
    m3.loadings.add(-loadings)[loadings.ne(0)]
            for m3 in m2
        ] for m2 in m1
    ] for m1 in models
]
ints_deltas = [[[
    m3.intercepts.add(-intercepts).reshape(-1)
            for m3 in m2
        ] for m2 in m1
    ] for m1 in models
]
cor_deltas = [[[
    m3.cov.add(-cor_mat)[cor_mat.tril(diagonal = -1).ne(0)]
            for m3 in m2
        ] for m2 in m1
    ] for m1 in models
]

ldgs_biases = [[
    torch.stack(d2, dim = 0).mean(dim = 0)
        for d2 in d1
    ] for d1 in ldgs_deltas
]
ldgs_mses = [[
    torch.stack(d2, dim = 0).pow(2).mean(dim = 0)
        for d2 in d1
    ] for d1 in ldgs_deltas
]
ldgs_bias_fig = boxplots(
    vals = ldgs_biases,
    sample_sizes = sample_sizes,
    y_label = "Bias",
    cond_labels = cond_labels,
    hline_locs = [0.],
)
ldgs_mse_fig = boxplots(
    vals = ldgs_mses,
    sample_sizes = sample_sizes,
    y_label = "MSE",
    cond_labels = cond_labels,
    log_scale = True,
)

ints_biases = [[
    torch.stack(d2, dim = 0).mean(dim = 0)
        for d2 in d1
    ] for d1 in ints_deltas
]
ints_mses = [[
    torch.stack(d2, dim = 0).pow(2).mean(dim = 0)
        for d2 in d1
    ] for d1 in ints_deltas
]
ints_bias_fig = boxplots(
    vals = ints_biases,
    sample_sizes = sample_sizes,
    y_label = "Bias",
    cond_labels = cond_labels,
    hline_locs = [0.],
)
ints_mse_fig = boxplots(
    vals = ints_mses,
    sample_sizes = sample_sizes,
    y_label = "MSE",
    cond_labels = cond_labels,
    log_scale = True,
)

cor_biases = [[
    torch.stack(d2, dim = 0).mean(dim = 0)
        for d2 in d1
    ] for d1 in cor_deltas
]
cor_mses = [[
    torch.stack(d2, dim = 0).pow(2).mean(dim = 0)
        for d2 in d1
    ] for d1 in cor_deltas
]
cor_bias_fig = boxplots(
    vals = cor_biases,
    sample_sizes = sample_sizes,
    y_label = "Bias",
    cond_labels = cond_labels,
    hline_locs = [0.],
)
cor_mse_fig = boxplots(
    vals = cor_mses,
    sample_sizes = sample_sizes,
    y_label = "MSE",
    cond_labels = cond_labels,
    log_scale = True,
)

for lab, fig in zip(
    ("ldgs", "ints", "cor"),
    (ldgs_bias_fig, ints_bias_fig, cor_bias_fig),
):
    pdf = matplotlib.backends.backend_pdf.PdfPages(
        fig_dir + "comparison_" + lab + "_bias_plt.pdf"
    )
    pdf.savefig(fig, dpi = 300)
    pdf.close()
    
for lab, fig in zip(
    ("ldgs", "ints", "cor"),
    (ldgs_mse_fig, ints_mse_fig, cor_mse_fig),
):
    pdf = matplotlib.backends.backend_pdf.PdfPages(
        fig_dir + "comparison_" + lab + "_mse_plt.pdf"
    )
    pdf.savefig(fig, dpi = 300)
    pdf.close()
    
time_fig = time_plt(
    models = models,
    sample_sizes = sample_sizes,
    cond_labels = cond_labels,
    legend_loc = "upper left",
    ylim = [0, 10**3]
)
pdf = matplotlib.backends.backend_pdf.PdfPages(
    fig_dir + "comparison_time_plt.pdf"
)
pdf.savefig(time_fig, dpi = 300)
pdf.close()

################################################################################
#
# Uniform C2ST figures
#
################################################################################

fig_dir = "figures/c2st/uniform/"
Path(fig_dir).mkdir(parents = True, exist_ok = True)

res_dir = "results/c2st/"

rr_types = ["t1_error", "power"]
sample_sizes = [250, 500, 1000, 2500, 5000, 10000]
n_reps = 100

c2sts = [[[
    load_obj(
        res_dir + rr_type +
        "/sample_size_" + str(i) +
        "/c2st_" + str(j)
            ) for j in range(n_reps)
        ] for i in range(len(sample_sizes))
    ] for rr_type in rr_types
]

rrs_t1 = torch.stack([
        torch.cat(
            [c2.p_val for c2 in c1], dim=0,
        ).lt(0.05).float().mean()
    for c1 in c2sts[0]], dim=0,
).unsqueeze(0)
accs_t1 = torch.stack([
        torch.stack(
            [c2.acc for c2 in c1], dim=0,
        ).quantile(torch.tensor([0.25, 0.5, 0.75]), dim=0)
    for c1 in c2sts[0]], dim=1,
)

rr_acc_t1_fig = rr_acc_plt(
    rrs = rrs_t1,
    accs = accs_t1,
    sample_sizes = sample_sizes,
    rr_ylim = [-0.005, 0.2],
    acc_ylim = [0.46, 1.],
    rr_hline = True,
    acc_line_loc = 0.525,
)
pdf = matplotlib.backends.backend_pdf.PdfPages(
    fig_dir + "rr_acc_t1.pdf"
)
pdf.savefig(rr_acc_t1_fig, dpi = 300)
pdf.close()

rrs_power = torch.stack([
        torch.cat(
            [c2.p_val for c2 in c1], dim=0,
        ).lt(0.05).float().mean()
    for c1 in c2sts[1]], dim=0,
)
rrs_power = torch.stack((
        rrs_power,
        torch.cat([
            approx_power(N = ss, eps = 0.025, delta = 0.025, alpha = 0.05)
            for ss in sample_sizes], dim = 0,
        )
    ), dim = 0,
)
accs_power = torch.stack([
        torch.stack(
            [c2.acc for c2 in c1], dim=0,
        ).quantile(torch.tensor([0.25, 0.5, 0.75]), dim=0)
    for c1 in c2sts[1]], dim=1,
)

rr_acc_power_fig = rr_acc_plt(
    rrs = rrs_power,
    accs = accs_power,
    sample_sizes = sample_sizes,
    rr_labels = ["Empirical Power", "Predicted Power"],
    rr_ylim = [-0.01, 1.01],
    acc_ylim = [0.46, 1.],
    acc_line_loc = 0.525,
    rr_trans = [-0.1, 0.1],
)
pdf = matplotlib.backends.backend_pdf.PdfPages(
    fig_dir + "rr_acc_power.pdf"
)
pdf.savefig(rr_acc_power_fig, dpi = 300)
pdf.close()

################################################################################
#
# I-WAVE C2ST figures
#
################################################################################

fig_dir = "figures/c2st/i-wave/"
Path(fig_dir).mkdir(parents = True, exist_ok = True)

res_dir = "results/c2st/"

model_types = ["five-factor", "seven-factor"]
sample_sizes = [625, 1250, 2500, 5000, 10000]
n_reps = 100

def approx_to_exact_p_val(c2st):
    sample_size = (
        c2st.global_fit_res["data_real_test"].shape[0] +
        c2st.global_fit_res["data_fake_test"].shape[0]
    )
    p_val = 1 - torch.distributions.Normal(
        loc = torch.tensor([0.5]),
        scale = torch.tensor([0.25 / sample_size]).sqrt()
    ).cdf(c2st.acc)
    
    return p_val

c2st_props = [[[
    load_obj(
        res_dir + "gen_" + mt1 +
        "/fit_" + mt2 +
        "/sample_size_" + str(i) +
        "/c2st_prop/c2st_prop_" + str(j)
            ) for j in range(n_reps)
        ] for i in range(len(sample_sizes))
    ] for mt1, mt2 in product(model_types, model_types)
]

rrs_gen_5_fit_7 = torch.stack([
        torch.stack([
            torch.cat(
                [approx_to_exact_p_val(c3) if i == 1
                 else c3.p_val for c3 in c2], dim=0,
            ).lt(0.05).float().mean()
        for c2 in c1], dim=0)
    for i, c1 in enumerate([c2st_props[1], c2st_props[1]])], dim=0,
)
accs_gen_5_fit_7 = torch.stack([
        torch.stack(
            [c2.acc for c2 in c1], dim=0,
        ).quantile(torch.tensor([0.25, 0.5, 0.75]), dim=0)
    for c1 in c2st_props[1]], dim=1,
)
rr_acc_gen_5_fit_7_fig = rr_acc_plt(
    rrs = rrs_gen_5_fit_7,
    accs = accs_gen_5_fit_7,
    sample_sizes = sample_sizes,
    rr_labels = ("Exact C2ST", "Approximate C2ST"),
    rr_hline = True,
    rr_ylim = [-0.005, 0.2],
    acc_ylim = [0.46, 1.],
)
pdf = matplotlib.backends.backend_pdf.PdfPages(
    fig_dir + "rr_acc_gen_5_fit_7.pdf"
)
pdf.savefig(rr_acc_gen_5_fit_7_fig, dpi = 300)
pdf.close()

rrs_gen_7_fit_5 = torch.stack([
        torch.stack([
            torch.cat(
                [approx_to_exact_p_val(c3) if i == 1
                 else c3.p_val for c3 in c2], dim=0,
            ).lt(0.05).float().mean()
        for c2 in c1], dim=0)
    for i, c1 in enumerate([c2st_props[2], c2st_props[2]])], dim=0,
)
accs_gen_7_fit_5 = torch.stack([
        torch.stack(
            [c2.acc for c2 in c1], dim=0,
        ).quantile(torch.tensor([0.25, 0.5, 0.75]), dim=0)
    for c1 in c2st_props[2]], dim=1,
)
rr_acc_gen_7_fit_5_fig = rr_acc_plt(
    rrs = rrs_gen_7_fit_5,
    accs = accs_gen_7_fit_5,
    sample_sizes = sample_sizes,
    rr_labels = ("Exact C2ST", "Approximate C2ST"),
    rr_legend_loc = "lower right",
    rr_bbox_to_anchor = (1, 0.1),
    rr_ylim = [-0.01, 1.01],
    acc_ylim = [0.46, 1.],
    frameon = True,
)
pdf = matplotlib.backends.backend_pdf.PdfPages(
    fig_dir + "rr_acc_gen_7_fit_5.pdf"
)
pdf.savefig(rr_acc_gen_7_fit_5_fig, dpi = 300)
pdf.close()

importances_gen_7_fit_5_fig = importances_plt(
    c2sts = c2st_props[2][-1],
    x_labels = ["Items 1–10",
                "Items 11–20",
                "Items 21–30",
                "Items 31–40",
                "Items 41–50"],
)
pdf = matplotlib.backends.backend_pdf.PdfPages(
    fig_dir + "importances_gen_5_fit_7_plt.pdf"
)
pdf.savefig(importances_gen_7_fit_5_fig, dpi = 300)
pdf.close()

time_fig = time_plt(
    models = c2st_props[1:3],
    sample_sizes = sample_sizes,
    cond_labels = ["Gen. = 5, Fit = 7", "Gen. = 7, Fit = 5"],
    ylim = [0, 300],
    x_trans = [-0.05, 0.05]
)
pdf = matplotlib.backends.backend_pdf.PdfPages(
    fig_dir + "c2st_prop_time_plt.pdf"
)
pdf.savefig(time_fig, dpi = 300)
pdf.close()

c2st_bases = [[[
    load_obj(
        res_dir + "gen_" + mt1 +
        "/fit_" + mt2 +
        "/sample_size_" + str(i) +
        "/c2st_base/c2st_base_" + str(j)
            ) for j in range(n_reps)
        ] for i in range(len(sample_sizes))
    ] for mt1, mt2 in product(model_types, model_types)
]

for cp, cb in zip(c2st_props, c2st_bases):
    for i in range(len(sample_sizes)):
        for j in range(n_reps):
            cp[i][j].timerecords["fit"] += cb[i][j].timerecords["fit"]

time_fig = time_plt(
    models = c2st_props[1:3],
    sample_sizes = sample_sizes,
    cond_labels = ["Gen. = 5, Fit = 7", "Gen. = 7, Fit = 5"],
    ylim = [0, 300],
    x_trans = [-0.05, 0.05]
)
pdf = matplotlib.backends.backend_pdf.PdfPages(
    fig_dir + "c2st-rfi_time_plt.pdf"
)
pdf.savefig(time_fig, dpi = 300)
pdf.close()

c2st_rfis = [[[
    load_obj(
        res_dir + "gen_" + mt1 +
        "/fit_" + mt2 +
        "/sample_size_" + str(i) +
        "/c2st_rfi/c2st_rfi_" + str(j)
            ) for j in range(n_reps)
        ] for i in range(len(sample_sizes))
    ] for mt1, mt2 in product(model_types, model_types)
]

c2st_rfi_fig = boxplots(
    vals = c2st_rfis[1:3],
    sample_sizes = sample_sizes,
    y_label = "C2ST-RFI",
    cond_labels = ["Gen. = 5, Fit = 7", "Gen. = 7, Fit = 5"],
    hline_locs = [1., 0.9],
    hline_linestyles = ["--", "-"]
)
pdf = matplotlib.backends.backend_pdf.PdfPages(
    fig_dir + "c2st-rfi_plt.pdf"
)
pdf.savefig(c2st_rfi_fig, dpi = 300)
pdf.close()