#!/usr/bin/env python
#
# Author: Christopher J. Urban
# Code Author: Christopher J. Urban
# Affil.: L. L. Thurstone Psychometric Laboratory in the
#         Dept. of Psychology and Neuroscience, UNC-Chapel Hill
# E-mail: cjurban@live.unc.edu
#
# Purpose: Functions creating figures.
#
###############################################################################

import torch
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
from pylab import setp
import matplotlib.patches as mpatches
from matplotlib.transforms import Affine2D
import seaborn as sns
from code.python.utils import *
from code.python.c2st import *

# Make bias boxplots for importance-weighting simulations.
def bias_boxplots(cell_res_ls,
                  gen_cor,
                  gen_loadings,
                  gen_intercepts,
                  sample_size_ls,
                  power = 1,
                  ldgs_lims = None,
                  cor_lims = None,
                  int_lims = None):
    """
    Args:
        cell_res_ls    (list of dict): List of dictionaries holding replication results.
        gen_cor        (Tensor): Factor correlation matrix.
        gen_loadings   (Tensor): Factor loadings matrix.
        gen_intercepts (Tensor): Vector of intercepts.
        sample_size_ls (list of int): List of sample sizes.
        power          (int): Power to raise biases to.
        ldgs_lims      (list): y-axis limits for loadings biases.
        cor_lims       (list): y-axis limits for factor correlation biases.
        int_lims       (list): y-axis limits for intercepts biases.
    """
    if power == 1:
        bias_type = "Bias"
    elif power == 2:
        bias_type = "MSE"
    x_ticks = np.arange(len(sample_size_ls) + 1).tolist()[1:]
    
    # Lists for saving results.
    loadings_bias_ls = []
    int_bias_ls      = []
    cor_bias_ls      = []
    
    # Compute biases.
    for cell_res in cell_res_ls:
        n_reps = len(cell_res["loadings"])
        loadings_bias_ls.append(reduce(np.add, [(invert_factors(ldgs) - invert_factors(gen_loadings)).sum(axis = 1)**power for
                                                ldgs in cell_res["loadings"]]).flatten() / n_reps)
        int_bias_ls.append(reduce(np.add, [(ints - gen_intercepts)**power for ints in cell_res["intercepts"]]).flatten() / n_reps)
        cor_biases = np.tril(reduce(np.add, [(invert_cor(cor, ldgs) - invert_cor(gen_cor, gen_loadings))**power for
                                             cor, ldgs in zip(cell_res["cor"], cell_res["loadings"])]) / n_reps, k = -1)
        cor_bias_ls.append(cor_biases[np.nonzero(cor_biases)])
        
    # Create boxplots.
    fig = plt.figure()
    fig.set_size_inches(8, 8, forward = True)
    gs = fig.add_gridspec(4, 4)
    gs.tight_layout(fig, rect = [0, 0.03, 1, 0.98])
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    ax2 = fig.add_subplot(gs[0:2, 2:])
    ax3 = fig.add_subplot(gs[2:, 1:3])
    
    plt.subplots_adjust(wspace = 1.1, hspace = 1.1)
    
    flierprops = dict(marker = "o", markerfacecolor = "black", markersize = 2, linestyle = "none")

    # Main loadings boxplots.
    b11 = ax1.boxplot(loadings_bias_ls[0:4],
                         positions = [0.5, 2.5, 4.5, 6.5],
                         widths = 0.45,
                         flierprops = flierprops,
                         patch_artist = True)
    for b in b11["medians"]:
        setp(b, color = "black")
    for b in b11["boxes"]:
        b.set(facecolor = "white")
    b12 = ax1.boxplot(loadings_bias_ls[4:8],
                         positions = [1, 3, 5, 7],
                         widths = 0.45,
                         flierprops = flierprops,
                         patch_artist = True)
    for b in b12["medians"]:
        setp(b, color = "black")
    for b in b12["boxes"]:
        b.set(facecolor = "gray")
    b13 = ax1.boxplot(loadings_bias_ls[8:12],
                         positions = [1.5, 3.5, 5.5, 7.5],
                         widths = 0.45,
                         flierprops = flierprops,
                         patch_artist = True)
    for b in b13["medians"]:
        setp(b, color = "white")
    for b in b13["boxes"]:
        b.set(facecolor = "black")
    ax1.set_xticks([1, 3, 5, 7])
    ax1.set_xticklabels(sample_size_ls)
    ax1.set_xlabel("Sample Size")
    if ldgs_lims is not None:
        ax1.set_ylim(ldgs_lims)
    ax1.set_ylabel(bias_type)
    if power == 1:
        ax1.axhline(y = 0, color = "lightgray", linestyle = "--")
    ax1.set_title("Loadings")
    
    # Set legend.
    circ1 = mpatches.Rectangle((0.5, 0.5), 1, 0.5, facecolor = "white", label = "1 IW Sample", edgecolor = "black")
    circ2 = mpatches.Rectangle((0.5, 0.5), 1, 0.5, facecolor = "gray", label = "5 IW Samples", edgecolor = "black")
    circ3 = mpatches.Rectangle((0.5, 0.5), 1, 0.5, facecolor = "black", label = "25 IW Samples", edgecolor = "black")
    if power == 1:
        ax1.legend(handles = [circ1, circ2, circ3], loc = "upper right", frameon = False, prop = {"size" : 8})
    else:
        ax1.legend(handles = [circ1, circ2, circ3], loc = "upper right", frameon = False, prop = {"size" : 8})
    
    # Factor correlation boxplots.
    b21 = ax2.boxplot(cor_bias_ls[0:4],
                         positions = [0.5, 2.5, 4.5, 6.5],
                         widths = 0.45,
                         flierprops = flierprops,
                         patch_artist = True)
    for b in b21["medians"]:
        setp(b, color = "black")
    for b in b21["boxes"]:
        b.set(facecolor = "white")
    b22 = ax2.boxplot(cor_bias_ls[4:8],
                         positions = [1, 3, 5, 7],
                         widths = 0.45,
                         flierprops = flierprops,
                         patch_artist = True)
    for b in b22["medians"]:
        setp(b, color = "black")
    for b in b22["boxes"]:
        b.set(facecolor = "gray")
    b23 = ax2.boxplot(cor_bias_ls[8:12],
                         positions = [1.5, 3.5, 5.5, 7.5],
                         widths = 0.45,
                         flierprops = flierprops,
                         patch_artist = True)
    for b in b23["medians"]:
        setp(b, color = "white")
    for b in b23["boxes"]:
        b.set(facecolor = "black")
    ax2.set_xticks([1, 3, 5, 7])
    ax2.set_xticklabels(sample_size_ls)
    ax2.set_xlabel("Sample Size")
    if cor_lims is not None:
        ax2.set_ylim(cor_lims)
    ax2.set_ylabel(bias_type)
    if power == 1:
        ax2.axhline(y = 0, color = "lightgray", linestyle = "--")
    ax2.set_title("Factor Correlations")    

    # Intercepts boxplots.
    b31 = ax3.boxplot(int_bias_ls[0:4],
                         positions = [0.5, 2.5, 4.5, 6.5],
                         widths = 0.45,
                         flierprops = flierprops,
                         patch_artist = True)
    for b in b31["medians"]:
        setp(b, color = "black")
    for b in b31["boxes"]:
        b.set(facecolor = "white")
    b32 = ax3.boxplot(int_bias_ls[4:8],
                         positions = [1, 3, 5, 7],
                         widths = 0.45,
                         flierprops = flierprops,
                         patch_artist = True)
    for b in b32["medians"]:
        setp(b, color = "black")
    for b in b32["boxes"]:
        b.set(facecolor = "gray")
    b33 = ax3.boxplot(int_bias_ls[8:12],
                         positions = [1.5, 3.5, 5.5, 7.5],
                         widths = 0.45,
                         flierprops = flierprops,
                         patch_artist = True)
    for b in b33["medians"]:
        setp(b, color = "white")
    for b in b33["boxes"]:
        b.set(facecolor = "black")
    ax3.set_xticks([1, 3, 5, 7])
    ax3.set_xticklabels(sample_size_ls)
    ax3.set_xlabel("Sample Size")
    if int_lims is not None:
        ax3.set_ylim(int_lims)
    ax3.set_ylabel(bias_type)
    if power == 1:
        ax3.axhline(y = 0, color = "lightgray", linestyle = "--")
    ax3.set_title("Intercepts")
    
    return fig

# Make time lineplots for importance-weighting simulations.
def time_plot(cell_res_ls,
              sample_size_ls,
              y_lims = None):
    """
    Args:
        cell_res_ls    (list of dict): List of dictionaries holding replication results.
        sample_size_ls (list of int): List of sample sizes.
        y_lims         (list of float): y-axis limits.
    """
    n_reps = len(cell_res_ls[0]["loadings"])
    x_ticks = np.arange(len(sample_size_ls) + 1).tolist()[1:]
        
    # Create subplots.
    fig, ax = plt.subplots(constrained_layout = True)
    fig.set_size_inches(5, 5, forward = True)
    
    trans0 = Affine2D().translate(+0.00, 0.0) + ax.transData
    trans1 = Affine2D().translate(-0.1, 0.0) + ax.transData
    trans2 = Affine2D().translate(+0.1, 0.0) + ax.transData
    
    qnts = [np.quantile(a, [.25, .50, .75]) for a in [cell_res_ls[0]["run_time"], cell_res_ls[1]["run_time"], cell_res_ls[2]["run_time"], cell_res_ls[3]["run_time"]]]
    meds = [p[1] for p in qnts]
    err1 = [abs(p[0] - p[1]) for p in qnts]
    err2 = [abs(p[1] - p[2]) for p in qnts]
    p1 = ax.errorbar(x_ticks, meds, yerr = [err1, err2], capsize = 5, fmt = "k-o", transform = trans0, markersize = 3, label = "1 IW Sample")
    qnts = [np.quantile(a, [.25, .50, .75]) for a in [cell_res_ls[4]["run_time"], cell_res_ls[5]["run_time"], cell_res_ls[6]["run_time"], cell_res_ls[7]["run_time"]]]
    meds = [p[1] for p in qnts]
    err1 = [abs(p[0] - p[1]) for p in qnts]
    err2 = [abs(p[1] - p[2]) for p in qnts]
    p2 = ax.errorbar(x_ticks, meds, yerr = [err1, err2], capsize = 5, fmt = "k--^", transform = trans1, markersize = 3, label = "5 IW Samples")
    qnts = [np.quantile(a, [.25, .50, .75]) for a in [cell_res_ls[8]["run_time"], cell_res_ls[9]["run_time"], cell_res_ls[10]["run_time"], cell_res_ls[11]["run_time"]]]
    meds = [p[1] for p in qnts]
    err1 = [abs(p[0] - p[1]) for p in qnts]
    err2 = [abs(p[1] - p[2]) for p in qnts]
    p3 = ax.errorbar(x_ticks, meds, yerr = [err1, err2], capsize = 5, fmt = "k:s", transform = trans2, markersize = 3, label = "25 IW Samples")
    handles, labels = ax.get_legend_handles_labels()
    handles = [h[0] for h in handles]
    ax.legend(handles, labels, loc = "upper right", frameon = False, prop = {"size" : 12}) 
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(sample_size_ls)
    ax.set_xlabel("Sample Size")
    if y_lims is not None:
        ax.set_ylim(y_lims)
    ax.set_ylabel("Fitting Time (Seconds)")
    
    return fig

# Make MSE boxplots for MH-RM comparisons.
def mhrm_boxplots(iwave_cell_res_ls,
                  mhrm_cell_res_ls,
                  gen_cor,
                  gen_loadings,
                  gen_intercepts,
                  sample_size_ls,
                  ldgs_lims = None,
                  cor_lims = None,
                  int_lims = None):
    """
    Args:
        iwave_cell_res_ls    (list of dict): List of dictionaries holding I-WAVE results.
        mhrm_cell_res_ls    (list of dict): List of dictionaries holding MH-RM results.
        gen_cor        (Tensor): Factor correlation matrix.
        gen_loadings   (Tensor): Factor loadings matrix.
        gen_intercepts (Tensor): Vector of intercepts.
        sample_size_ls (list of int): List of sample sizes.
        ldgs_lims      (list): y-axis limits for loadings biases.
        cor_lims       (list): y-axis limits for factor correlation biases.
        int_lims       (list): y-axis limits for intercepts biases.
    """
    x_ticks = np.arange(len(sample_size_ls) + 1).tolist()[1:]
    
    # Lists for saving results.
    iwave_loadings_bias_ls = []
    iwave_int_bias_ls      = []
    iwave_cor_bias_ls      = []
    mhrm_loadings_bias_ls  = []
    mhrm_int_bias_ls       = []
    mhrm_cor_bias_ls       = []
    
    # Compute biases.
    for cell_res in iwave_cell_res_ls:
        n_reps = len(cell_res["loadings"])
        iwave_loadings_bias_ls.append(reduce(np.add, [(invert_factors(ldgs) - invert_factors(gen_loadings.numpy())).sum(axis = 1)**2 for
                                                      ldgs in cell_res["loadings"]]).flatten() / n_reps)
        iwave_int_bias_ls.append(reduce(np.add, [(ints - gen_intercepts.numpy())**2 for ints, ldgs in
                                                 zip(cell_res["intercepts"], cell_res["loadings"])]).flatten() / n_reps)
        cor_biases = np.tril(reduce(np.add, [(invert_cor(cor, ldgs) - invert_cor(gen_cor.numpy(), gen_loadings.numpy()))**2 for
                                             cor, ldgs in zip(cell_res["cor"], cell_res["loadings"])]) / n_reps, k = -1)
        iwave_cor_bias_ls.append(cor_biases[np.nonzero(cor_biases)])
    for cell_res in mhrm_cell_res_ls:
        n_reps = len(cell_res["loadings"])
        mhrm_loadings_bias_ls.append(reduce(np.add, [(invert_factors(ldgs) - invert_factors(gen_loadings.numpy())).sum(axis = 1)**2 for
                                                      ldgs in cell_res["loadings"]]).flatten() / n_reps)
        
        # Handle missing intercepts.
        ints_masks = [(np.isnan(ints - gen_intercepts.numpy()) * 1 - 1) * -1 for ints in cell_res["intercepts"]]
        ints_reps = reduce(np.add, ints_masks)
        ints_ls = cell_res["intercepts"].copy()
        for ints in ints_ls:
            ints[np.isnan(ints)] = 0
        mhrm_int_bias_ls.append(reduce(np.add, [(ints - gen_intercepts.numpy())**2 for ints, ldgs in
                                                zip(ints_ls, cell_res["loadings"])]).flatten() / ints_reps)

        cor_biases = np.tril(reduce(np.add, [(invert_cor(cor, ldgs) - invert_cor(gen_cor.numpy(), gen_loadings.numpy()))**2 for
                                             cor, ldgs in zip(cell_res["cor"], cell_res["loadings"])]) / n_reps, k = -1)
        mhrm_cor_bias_ls.append(cor_biases[np.nonzero(cor_biases)])
        
    # Create boxplots.
    fig = plt.figure()
    fig.set_size_inches(8, 8, forward = True)
    gs = fig.add_gridspec(4, 4)
    gs.tight_layout(fig, rect = [0, 0.03, 1, 0.98])
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    ax2 = fig.add_subplot(gs[0:2, 2:])
    ax3 = fig.add_subplot(gs[2:, 1:3])
    
    plt.subplots_adjust(wspace = 1.1, hspace = 1.1)
    flierprops = dict(marker = "o", markerfacecolor = "black", markersize = 2, linestyle = "none")

    # Main loadings boxplots.
    b11 = ax1.boxplot(iwave_loadings_bias_ls,
                      positions = [0.5, 2.5, 4.5, 6.5],
                      widths = 0.45,
                      flierprops = flierprops,
                      patch_artist = True)
    for b in b11["medians"]:
        setp(b, color = "black")
    for b in b11["boxes"]:
        b.set(facecolor = "white")
    b12 = ax1.boxplot(mhrm_loadings_bias_ls,
                      positions = [1, 3, 5, 7],
                      widths = 0.45,
                      flierprops = flierprops,
                      patch_artist = True)
    for b in b12["medians"]:
        setp(b, color = "white")
    for b in b12["boxes"]:
        b.set(facecolor = "black")
    ax1.set_xticks([0.75, 2.75, 4.75, 6.75])
    ax1.set_xticklabels(sample_size_ls)
    ax1.set_xlabel("Sample Size")
    if ldgs_lims is not None:
        ax1.set_ylim(ldgs_lims)
    ax1.set_ylabel("MSE")
    ax1.set_title("Loadings")
    
    # Set legend.
    circ1 = mpatches.Rectangle((0.5, 0.5), 1, 0.5, facecolor = "white", label = "I-WAVE", edgecolor = "black")
    circ2 = mpatches.Rectangle((0.5, 0.5), 1, 0.5, facecolor = "black", label = "MH-RM", edgecolor = "black")
    ax1.legend(handles = [circ1, circ2], loc = "upper right", frameon = False, prop = {"size" : 10})
    
    # Factor correlation boxplots.
    b21 = ax2.boxplot(iwave_cor_bias_ls,
                      positions = [0.5, 2.5, 4.5, 6.5],
                      widths = 0.45,
                      flierprops = flierprops,
                      patch_artist = True)
    for b in b21["medians"]:
        setp(b, color = "black")
    for b in b21["boxes"]:
        b.set(facecolor = "white")
    b22 = ax2.boxplot(mhrm_cor_bias_ls,
                      positions = [1, 3, 5, 7],
                      widths = 0.45,
                      flierprops = flierprops,
                      patch_artist = True)
    for b in b22["medians"]:
        setp(b, color = "white")
    for b in b22["boxes"]:
        b.set(facecolor = "black")
    ax2.set_xticks([0.75, 2.75, 4.75, 6.75])
    ax2.set_xticklabels(sample_size_ls)
    ax2.set_xlabel("Sample Size")
    if cor_lims is not None:
        ax2.set_ylim(cor_lims)
    ax2.set_ylabel("MSE")
    ax2.set_title("Factor Correlations")    

    # Intercepts boxplots.
    b31 = ax3.boxplot(iwave_int_bias_ls,
                      positions = [0.5, 2.5, 4.5, 6.5],
                      widths = 0.45,
                      flierprops = flierprops,
                      patch_artist = True)
    for b in b31["medians"]:
        setp(b, color = "black")
    for b in b31["boxes"]:
        b.set(facecolor = "white")
    b32 = ax3.boxplot(mhrm_int_bias_ls,
                      positions = [1, 3, 5, 7],
                      widths = 0.45,
                      flierprops = flierprops,
                      patch_artist = True)
    for b in b32["medians"]:
        setp(b, color = "white")
    for b in b32["boxes"]:
        b.set(facecolor = "black")
    ax3.set_xticks([0.75, 2.75, 4.75, 6.75])
    ax3.set_xticklabels(sample_size_ls)
    ax3.set_xlabel("Sample Size")
    if int_lims is not None:
        ax3.set_ylim(int_lims)
    ax3.set_ylabel("MSE")
    ax3.set_title("Intercepts")
    
    return fig

# Make line plots of fitting times for comparison studies.
def comparison_time_plots(iwave_cell_res_ls,
                          comp_cell_res_ls,
                          sample_size_ls,
                          lab1,
                          lab2,
                          y_lims = None):
    """
    Args:
        iwave_cell_res_ls (list of dict): List of dictionaries holding I-WAVE results.
        comp_cell_res_ls  (list of dict): List of dictionaries holding comparison method results.      
    """
    x_ticks = np.arange(len(iwave_cell_res_ls) + 1).tolist()[1:]
        
    # Create subplots.
    fig, ax = plt.subplots(constrained_layout = True)
    fig.set_size_inches(5, 5, forward = True)
    
    trans0 = Affine2D().translate(+0.00, 0.0) + ax.transData
    trans1 = Affine2D().translate(+0.00, 0.0) + ax.transData
    
    qnts = [np.quantile(a, [.25, .50, .75]) for a in [cell_res["run_time"] for cell_res in iwave_cell_res_ls]]
    meds = [p[1] for p in qnts]
    err1 = [abs(p[0] - p[1]) for p in qnts]
    err2 = [abs(p[1] - p[2]) for p in qnts]
    p1 = ax.errorbar(x_ticks, meds, yerr = [err1, err2], capsize = 5, fmt = "k-o", transform = trans0, markersize = 3, label = lab1)
    qnts = [np.quantile(a, [.25, .50, .75]) for a in [cell_res["run_time"] for cell_res in comp_cell_res_ls]]
    meds = [p[1] for p in qnts]
    err1 = [abs(p[0] - p[1]) for p in qnts]
    err2 = [abs(p[1] - p[2]) for p in qnts]
    p2 = ax.errorbar(x_ticks, meds, yerr = [err1, err2], capsize = 5, fmt = "k--^", transform = trans1, markersize = 3, label = lab2)
    handles, labels = ax.get_legend_handles_labels()
    handles = [h[0] for h in handles]
    ax.legend(handles, labels, loc = "upper left", frameon = False, prop = {"size" : 12})
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(sample_size_ls)
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Fitting Time (Seconds)")
    if y_lims is not None:
        ax.set_ylim(y_lims)
    
    return fig

# Make rejection rate plot.
def rr_plot(knn_p_val_ls_ls,
            nn_p_val_ls_ls,
            sample_size_ls,
            plot_line = False,
            rr_lim = None,
            rr_trans = False,
            legend_loc = "upper left"):
    """
    Args:
        knn_p_val_ls_ls  (list of list): List of lists of KNN p-values.
        nn_p_val_ls_ls   (list of list): List of lists of NN p-values.
        sample_size_ls   (list of int): List of sample sizes.
        plot_line        (Boolean): Whether to put a dotted line at RR = 0.2.
        rr_lim           (list of int): y-axis limits for RR plot.
        legend_loc       (str): Location of legend.
    """
    n_cells = len(sample_size_ls)
    
    # Compute rejection rates.
    knn_rr = [np.mean([p < 0.05 for p in ls]) for ls in knn_p_val_ls_ls]
    nn_rr = [np.mean([p < 0.05 for p in ls]) for ls in nn_p_val_ls_ls]
    
    # Set plot layout.
    fig, ax = plt.subplots(constrained_layout = True)
    fig.set_size_inches(5, 5, forward = True)
    fig.set_size_inches(5, 5, forward = True)
    
    # Transformations to shift line plots horizontally.
    if rr_trans:
        trans0 = Affine2D().translate(-0.1, 0.0) + ax.transData
        trans1 = Affine2D().translate(+0.1, 0.0) + ax.transData
    else:
        trans0 = Affine2D().translate(0.0, 0.0) + ax.transData
        trans1 = Affine2D().translate(0.0, 0.0) + ax.transData
    
    # Make plot.
    if plot_line:
        ax.axhline(0.05, linestyle = "dashed", color = "lightgray")
    ax.plot(np.arange(1, n_cells + 1).tolist(), knn_rr, "k-o", transform = trans0, label = "KNN")
    ax.plot(np.arange(1, n_cells + 1).tolist(), nn_rr, "k:^", transform = trans1, label = "NN")
    ax.legend(loc = legend_loc, frameon = False, prop = {"size" : 12}) 
    ax.set_xticks(np.arange(1, n_cells + 1).tolist())
    ax.set_xticklabels(sample_size_ls)
    if rr_lim is not None:
        ax.set_ylim(rr_lim)
    ax.set_ylabel("Null Rejection Rate")
    ax.set_xlabel("Test Set Size")
    
    return fig

# Make rejection rate and accuracy plot.
def rr_acc_plot(knn_p_val_ls_ls,
                nn_p_val_ls_ls,
                knn_acc_ls_ls,
                nn_acc_ls_ls,
                sample_size_ls,
                guide_p_val_ls = None,
                plot_line = False,
                rr_lim = None,
                acc_lim = None,
                rr_legend_loc = "upper left",
                acc_legend_loc = "upper left",
                rr_legend_size = 10,
                acc_legend_size = 10,
                acc_line_loc = 0.5,
                rr_trans = False,
                acc_trans = False):
    """
    Args:
        knn_p_val_ls_ls  (list of list): List of lists of KNN p-values.
        nn_p_val_ls_ls   (list of list): List of lists of NN p-values.
        knn_acc_ls_ls    (list of list): List of lists of KNN accuracies.
        nn_acc_ls_ls     (list of list): List of lists of NN accuracies.
        sample_size_ls   (list of int): List of sample sizes.
        guide_p_val_ls   (list of int): List of guide p-values.
        plot_line        (Boolean): Whether to put a dotted line at RR = 0.2.
        rr_lim           (list of int): y-axis limits for RR plot.
        acc_lim          (list of int): y-axis limits for accuracy plot.
        rr_legend_loc    (str): Location of legend for RR plot.
        acc_legend_loc   (str): Location of legend for accuracy plot.
        acc_line_loc     (float): Location of line on accuracy plot.
        rr_trans         (Boolean): Whether to translate RR line plots.
        acc_trans        (Boolean): Whether to translate accuracy line plots.
    """
    x_ticks = np.arange(len(sample_size_ls) + 1).tolist()[1:]
    
    # Compute rejection rates.
    knn_rr = [np.mean([p < 0.05 for p in ls]) for ls in knn_p_val_ls_ls]
    nn_rr = [np.mean([p < 0.05 for p in ls]) for ls in nn_p_val_ls_ls]
    
    # Set plot layout.
    fig = plt.figure()
    fig.set_size_inches(12, 5, forward = True)
    gs = fig.add_gridspec(2, 4)
    gs.tight_layout(fig, rect = [0, 0.03, 1, 0.98])
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    ax2 = fig.add_subplot(gs[0:2, 2:])
    plt.subplots_adjust(wspace = 1.1, hspace = 0.05)

    # Transformations to shift line plots horizontally.
    if guide_p_val_ls is not None:
        trans0 = Affine2D().translate(-0.2, 0.0) + ax1.transData
        trans1 = Affine2D().translate(+0.0, 0.0) + ax1.transData
        trans4 = Affine2D().translate(+0.2, 0.0) + ax1.transData
    elif rr_trans:
        trans0 = Affine2D().translate(-0.1, 0.0) + ax1.transData
        trans1 = Affine2D().translate(+0.1, 0.0) + ax1.transData
    else:
        trans0 = Affine2D().translate(0.0, 0.0) + ax1.transData
        trans1 = Affine2D().translate(0.0, 0.0) + ax1.transData
    if acc_trans:
        trans2 = Affine2D().translate(-0.1, 0.0) + ax2.transData
        trans3 = Affine2D().translate(+0.1, 0.0) + ax2.transData
    else:
        trans2 = Affine2D().translate(0.0, 0.0) + ax2.transData
        trans3 = Affine2D().translate(0.0, 0.0) + ax2.transData

    # Make rejection rate line plot.
    if plot_line:
        ax1.axhline(0.05, linestyle = "dashed", color = "lightgray")
    if guide_p_val_ls is not None:
        ax1.plot(x_ticks, guide_p_val_ls, "k:^", transform = trans4, label = "Predicted Power",
                 ls = "-", marker = "+", c = "gray")
    ax1.plot(x_ticks, knn_rr, "k-o", transform = trans0, label = "KNN")
    ax1.plot(x_ticks, nn_rr, "k:^", transform = trans1, label = "NN")
    ax1.legend(loc = rr_legend_loc, frameon = False, prop = {"size" : rr_legend_size})
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(sample_size_ls)
    ax1.set_ylabel("Null Rejection Rate")
    ax1.set_xlabel("Test Set Size")
    if rr_lim is not None:
        ax1.set_ylim(rr_lim)
        
    # Make accuracy line plot.
    ax2.axhline(acc_line_loc, linestyle = "dashed", color = "lightgray", zorder = 1)
    qnts = [np.quantile(a, [.25, .50, .75]) for a in knn_acc_ls_ls]
    meds = [p[1] for p in qnts]
    err1 = [abs(p[0] - p[1]) for p in qnts]
    err2 = [abs(p[1] - p[2]) for p in qnts]
    p1 = ax2.errorbar(x_ticks, meds, yerr = [err1, err2], capsize = 5, fmt = "k-o",
                      transform = trans2, markersize = 3, label = "KNN", zorder = 2)
    qnts = [np.quantile(a, [.25, .50, .75]) for a in nn_acc_ls_ls]
    meds = [p[1] for p in qnts]
    err1 = [abs(p[0] - p[1]) for p in qnts]
    err2 = [abs(p[1] - p[2]) for p in qnts]
    p2 = ax2.errorbar(x_ticks, meds, yerr = [err1, err2], capsize = 5, fmt = "k:^",
                      transform = trans3, markersize = 3, label = "NN", zorder = 3)
    handles, labels = ax2.get_legend_handles_labels()
    handles = [h[0] for h in handles]
    ax2.legend(handles, labels, loc = acc_legend_loc, frameon = False, prop = {"size" : acc_legend_size}) 
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels(sample_size_ls)
    ax2.set_ylabel("Test Set Classification Accuracy")
    ax2.set_xlabel("Test Set Size")
    if acc_lim is not None:
        ax2.set_ylim(acc_lim)
    
    return fig

# Make rejection rate and accuracy plot with many C2STs.
def mul_rr_acc_plots(knn_p_val_res_ls,
                     nn_p_val_res_ls,
                     knn_acc_ls_ls,
                     nn_acc_ls_ls,
                     sample_size_ls,
                     plot_line = False,
                     rr_lim = None,
                     acc_lim = None,
                     rr_legend_loc = "upper left",
                     acc_legend_loc = "upper left",
                     rr_legend_size = 10,
                     acc_legend_size = 10,
                     acc_line_loc = 0.5,
                     rr_trans = False,
                     acc_trans = False):
    """
    Args:
        knn_p_val_res_ls (list of list): List holding KNN-based C2ST-RFI results.
        nn_p_val_res_ls  (list of list): List holding neural network-based C2ST-RFI results.
        knn_acc_ls_ls    (list of list): List of lists of KNN accuracies.
        nn_acc_ls_ls     (list of list): List of lists of neural network accuracies.
        sample_size_ls   (list of int): List of sample sizes.
        plot_line        (Boolean): Whether to put a dotted line at RR = 0.2.
        rr_lim           (list of int): y-axis limits for RR plot.
        acc_lim          (list of int): y-axis limits for accuracy plot.
        rr_legend_loc    (str): Location of legend for RR plot.
        acc_legend_loc   (str): Location of legend for accuracy plot.
        acc_line_loc     (float): Location of line on accuracy plot.
        rr_trans         (Boolean): Whether to translate RR line plots.
        acc_trans        (Boolean): Whether to translate accuracy line plots.
    """
    x_ticks = np.arange(len(sample_size_ls) + 1).tolist()[1:]
    
    # Compute rejection rates.
    knn_approx_rr = [np.mean([p < 0.05 for p in ls]) for ls in knn_p_val_res_ls[0]]
    nn_approx_rr = [np.mean([p < 0.05 for p in ls]) for ls in nn_p_val_res_ls[0]]
    knn_exact_rr = [np.mean([p < 0.05 for p in ls]) for ls in knn_p_val_res_ls[1]]
    nn_exact_rr = [np.mean([p < 0.05 for p in ls]) for ls in nn_p_val_res_ls[1]]
    
    # Set plot layout.
    fig = plt.figure()
    fig.set_size_inches(12, 5, forward = True)
    gs = fig.add_gridspec(2, 4)
    gs.tight_layout(fig, rect = [0, 0.03, 1, 0.98])
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    ax2 = fig.add_subplot(gs[0:2, 2:])
    plt.subplots_adjust(wspace = 1.1, hspace = 0.05)
    
    # Transformations to shift line plots horizontally.
    if rr_trans:
        trans0 = Affine2D().translate(-0.2, 0.0) + ax1.transData
        trans1 = Affine2D().translate(-0.067, 0.0) + ax1.transData
        trans2 = Affine2D().translate(+0.067, 0.0) + ax1.transData
        trans3 = Affine2D().translate(+0.2, 0.0) + ax1.transData
    else:
        trans0 = Affine2D().translate(0.0, 0.0) + ax1.transData
        trans1 = Affine2D().translate(0.0, 0.0) + ax1.transData
        trans2 = Affine2D().translate(0.0, 0.0) + ax1.transData
        trans3 = Affine2D().translate(0.0, 0.0) + ax1.transData
    if acc_trans:
        trans4 = Affine2D().translate(-0.1, 0.0) + ax2.transData
        trans5 = Affine2D().translate(+0.1, 0.0) + ax2.transData
    else:
        trans4 = Affine2D().translate(0.0, 0.0) + ax2.transData
        trans5 = Affine2D().translate(0.0, 0.0) + ax2.transData

    # Make rejection rate line plot.
    if plot_line:
        ax1.axhline(0.05, linestyle = "dashed", color = "lightgray")
    ax1.plot(x_ticks, knn_approx_rr, "k-o", transform = trans0, label = "KNN-Based Approx. C2ST")
    ax1.plot(x_ticks, nn_approx_rr, "k:^", transform = trans1, label = "NN-Based Approx. C2ST")
    ax1.plot(x_ticks, knn_exact_rr, transform = trans2, alpha = 0.75,
             ls = "-", marker = "o", c = "gray", label = "KNN-Based Exact C2ST")
    ax1.plot(x_ticks, nn_exact_rr, transform = trans3, alpha = 0.75,
             ls = ":", marker = "^", c = "gray", label = "NN-Based Exact C2ST")
    ax1.legend(loc = rr_legend_loc, frameon = False, prop = {"size" : rr_legend_size})
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(sample_size_ls)
    ax1.set_ylabel("Null Rejection Rate")
    ax1.set_xlabel("Test Set Size")
    if rr_lim is not None:
        ax1.set_ylim(rr_lim)
        
    # Make accuracy line plot.
    ax2.axhline(acc_line_loc, linestyle = "dashed", color = "lightgray", zorder = 1)
    qnts = [np.quantile(a, [.25, .50, .75]) for a in knn_acc_ls_ls]
    meds = [p[1] for p in qnts]
    err1 = [abs(p[0] - p[1]) for p in qnts]
    err2 = [abs(p[1] - p[2]) for p in qnts]
    p1 = ax2.errorbar(x_ticks, meds, yerr = [err1, err2], capsize = 5, fmt = "k-o",
                      transform = trans4, markersize = 3, label = "KNN", zorder = 2)
    qnts = [np.quantile(a, [.25, .50, .75]) for a in nn_acc_ls_ls]
    meds = [p[1] for p in qnts]
    err1 = [abs(p[0] - p[1]) for p in qnts]
    err2 = [abs(p[1] - p[2]) for p in qnts]
    p2 = ax2.errorbar(x_ticks, meds, yerr = [err1, err2], capsize = 5, fmt = "k:^",
                      transform = trans5, markersize = 3, label = "NN", zorder = 3)
    handles, labels = ax2.get_legend_handles_labels()
    handles = [h[0] for h in handles]
    ax2.legend(handles, labels, loc = acc_legend_loc, frameon = False, prop = {"size" : acc_legend_size}) 
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels(sample_size_ls)
    ax2.set_ylabel("Test Set Classification Accuracy")
    ax2.set_xlabel("Test Set Size")
    if acc_lim is not None:
        ax2.set_ylim(acc_lim)
    
    return fig

# Make C2ST-RFI boxplots.
def c2st_rfi_boxplots(knn_rfi_res_ls,
                      nn_rfi_res_ls,
                      sample_size_ls,
                      abs_lims = None,
                      max_lims = None,
                      id_lims = None):
    """
    Args:
        knn_rfi_res_ls (list of list): List holding KNN-based C2ST-RFI results.
        nn_rfi_res_ls  (list of list): List holding neural network-based C2ST-RFI results.
        sample_size_ls (list of int): List of sample sizes.
        abs_lims       (list of int): y-axis limits for C2ST-RFI with absolute value function.
        max_lims       (list of int): y-axis limits for C2ST-RFI with max function.
        id_lims        (list of int): y-axis limits for C2ST-RFI with identity function.
    """
    x_ticks = np.arange(len(sample_size_ls) + 1).tolist()[1:]
        
    # Create boxplots.
    fig = plt.figure()
    fig.set_size_inches(8, 8, forward = True)
    gs = fig.add_gridspec(4, 4)
    gs.tight_layout(fig, rect = [0, 0.03, 1, 0.98])
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    ax2 = fig.add_subplot(gs[0:2, 2:])
    ax3 = fig.add_subplot(gs[2:, 1:3])
    plt.subplots_adjust(wspace = 1.1, hspace = 1.1)
    
    # Mark perfect fit with dotted line.
    ax1.axhline(y = 1, color = "lightgray", linestyle = "--")
    ax2.axhline(y = 1, color = "lightgray", linestyle = "--")
    ax3.axhline(y = 1, color = "lightgray", linestyle = "--")
    
    # Set outlier marker style.
    flierprops = dict(marker = "o", markerfacecolor = "black", markersize = 2, linestyle = "none")

    # C2ST-RFI_abs boxplots.
    b11 = ax1.boxplot(knn_rfi_res_ls[0],
                      positions = [0.5, 2.5, 4.5, 6.5, 8.5],
                      widths = 0.45,
                      flierprops = flierprops,
                      patch_artist = True)
    for b in b11["medians"]:
        setp(b, color = "black")
    for b in b11["boxes"]:
        b.set(facecolor = "white")
    b12 = ax1.boxplot(nn_rfi_res_ls[0],
                      positions = [1, 3, 5, 7, 9],
                      widths = 0.45,
                      flierprops = flierprops,
                      patch_artist = True)
    for b in b12["medians"]:
        setp(b, color = "white")
    for b in b12["boxes"]:
        b.set(facecolor = "black")
    ax1.set_xticks([0.75, 2.75, 4.75, 6.75, 8.75])
    ax1.set_xticklabels(sample_size_ls)
    ax1.set_xlabel("Test Set Size")
    if abs_lims is not None:
        ax1.set_ylim(abs_lims)
    ax1.set_ylabel("C2ST-$\mathregular{RFI}_{\mathregular{abs}}$")
    
    # Set legend on first plot.
    circ1 = mpatches.Rectangle((0.5, 0.5), 1, 0.5, facecolor = "white", label = "KNN", edgecolor = "black")
    circ2 = mpatches.Rectangle((0.5, 0.5), 1, 0.5, facecolor = "black", label = "NN", edgecolor = "black")
    ax1.legend(handles = [circ1, circ2], loc = "lower right", frameon = False, prop = {"size" : 10})
    
    # C2ST-RFI_max boxplots.
    b21 = ax2.boxplot([[rfi for rfi in rfi_ls if not np.isnan(rfi)] for rfi_ls in knn_rfi_res_ls[1]],
                      positions = [0.5, 2.5, 4.5, 6.5, 8.5],
                      widths = 0.45,
                      flierprops = flierprops,
                      patch_artist = True)
    for b in b21["medians"]:
        setp(b, color = "white")
    setp(b21["medians"][4], color = "black")
    for b in b21["boxes"]:
        b.set(facecolor = "white")
    b22 = ax2.boxplot([[rfi for rfi in rfi_ls if not np.isnan(rfi)] for rfi_ls in nn_rfi_res_ls[1]],
                      positions = [1, 3, 5, 7, 9],
                      widths = 0.45,
                      flierprops = flierprops,
                      patch_artist = True)
    for b in b22["medians"]:
        setp(b, color = "black")
    setp(b22["medians"][4], color = "white")
    for b in b22["boxes"]:
        b.set(facecolor = "black")
    ax2.set_xticks([0.75, 2.75, 4.75, 6.75, 8.75])
    ax2.set_xticklabels(sample_size_ls)
    ax2.set_xlabel("Test Set Size")
    if max_lims is not None:
        ax2.set_ylim(max_lims)
    ax2.set_ylabel("C2ST-$\mathregular{RFI}_{\mathregular{max}}$")    

    # C2ST-RFI_id boxplots.
    b31 = ax3.boxplot(knn_rfi_res_ls[2],
                      positions = [0.5, 2.5, 4.5, 6.5, 8.5],
                      widths = 0.45,
                      flierprops = flierprops,
                      patch_artist = True)
    for b in b31["medians"]:
        setp(b, color = "black")
    for b in b31["boxes"]:
        b.set(facecolor = "white")
    b32 = ax3.boxplot(nn_rfi_res_ls[2],
                      positions = [1, 3, 5, 7, 9],
                      widths = 0.45,
                      flierprops = flierprops,
                      patch_artist = True)
    for b in b32["medians"]:
        setp(b, color = "white")
    for b in b32["boxes"]:
        b.set(facecolor = "black")
    ax3.set_xticks([0.75, 2.75, 4.75, 6.75, 8.75])
    ax3.set_xticklabels(sample_size_ls)
    ax3.set_xlabel("Test Set Size")
    if id_lims is not None:
        ax3.set_ylim(id_lims)
    ax3.set_ylabel("C2ST-$\mathregular{RFI}_{\mathregular{id}}$")
    
    return fig

# Make a single C2ST-RFI boxplot.
def c2st_rfi_boxplot(knn_rfi_res,
                     nn_rfi_res,
                     sample_size_ls,
                     lims = None):
    """
    Args:
        knn_rfi_res_ls (list of list): List holding KNN-based C2ST-RFI results.
        nn_rfi_res_ls  (list of list): List holding neural network-based C2ST-RFI results.
        sample_size_ls (list of int): List of sample sizes.
        abs_lims       (list of int): y-axis limits for C2ST-RFI with absolute value function.
        correct_spec   (Boolean): Whether or not IFA model is correctly specified.
        max_lims       (list of int): y-axis limits for C2ST-RFI with max function.
        id_lims        (list of int): y-axis limits for C2ST-RFI with identity function.
    """
    x_ticks = np.arange(len(sample_size_ls) + 1).tolist()[1:]
        
    # Create subplots.
    fig, ax = plt.subplots(constrained_layout = True)
    fig.set_size_inches(5, 5, forward = True)
    
    ax.axhline(y = 1, color = "lightgray", linestyle = "--")
    ax.axhline(y = 0.9, color = "lightgray", linestyle = "-")
    
    # Set outlier marker style.
    flierprops = dict(marker = "o", markerfacecolor = "black", markersize = 2, linestyle = "none")

    # C2ST-RFI_abs boxplots.
    b1 = ax.boxplot(knn_rfi_res,
                    positions = [0.5, 2.5, 4.5, 6.5, 8.5],
                    widths = 0.45,
                    flierprops = flierprops,
                    patch_artist = True)
    for b in b1["medians"]:
        setp(b, color = "black")
    for b in b1["boxes"]:
        b.set(facecolor = "white")
    b2 = ax.boxplot(nn_rfi_res,
                    positions = [1, 3, 5, 7, 9],
                    widths = 0.45,
                    flierprops = flierprops,
                    patch_artist = True)
    for b in b2["medians"]:
        setp(b, color = "white")
    for b in b2["boxes"]:
        b.set(facecolor = "black")
    ax.set_xticks([0.75, 2.75, 4.75, 6.75, 8.75])
    ax.set_xticklabels(sample_size_ls)
    ax.set_xlabel("Test Set Size")
    if lims is not None:
        ax.set_ylim(lims)
    ax.set_yticks(list(ax.get_yticks()) + [0.9])
    ax.set_ylabel("C2ST-RFI")
    
    # Set legend on first plot.
    circ1 = mpatches.Rectangle((0.5, 0.5), 1, 0.5, facecolor = "white", label = "KNN", edgecolor = "black")
    circ2 = mpatches.Rectangle((0.5, 0.5), 1, 0.5, facecolor = "black", label = "NN", edgecolor = "black")
    ax.legend(handles = [circ1, circ2], loc = "upper right", frameon = False, prop = {"size" : 12})
    
    return fig

# Make bar plot of variable importances.
def importance_plot(knn_imp_ls,
                    nn_imp_ls,
                    varnames,
                    knn_title,
                    nn_title,
                    knn_ylim = None,
                    nn_ylim = None,
                    hatch_list = None):
    """
    Args:
        knn_imp_ls (list): List of KNN variable importances.
        nn_imp_ls  (list): List of NN variable importances.
        varnames   (list of str): List of names of factors.
    """
    fig = plt.figure()
    fig.set_size_inches(8, 6, forward = True)
    
    # Create subplots.
    gs = fig.add_gridspec(8, 8)
    gs.tight_layout(fig, rect = [0, 0.03, 1, 0.98])
    ax1 = fig.add_subplot(gs[0:2, :])
    ax2 = fig.add_subplot(gs[3:, :])
    plt.subplots_adjust(wspace = 1.1, hspace = 1.1)
    
    # Get permutation importances.
    knn_imp_mean_ls = [knn_imp.importances_mean for knn_imp in knn_imp_ls]
    nn_imp_mean_ls  = [nn_imp.importances_mean for nn_imp in nn_imp_ls]

    # Set bar widths and locations.
    groups = len(varnames)
    x_ticks = np.arange(groups)
    width = 1 / 12
    bar_locs = np.concatenate([-np.arange(1, 10, 2)[::-1],
                               np.arange(1, 10, 2)]) * 0.5 * (8 / 8) * width
    
    # Make bar plots.
    ranges = np.arange(0, 51, 10).tolist()
    for i, j in zip(ranges[:-1], ranges[1:]):
        qnts = np.quantile(knn_imp_mean_ls, [.25, .5, .75], axis = 0)[:, i:j]
        importances_mean_meds = qnts[1, :]
        qnts = np.concatenate([np.abs(qnts[np.newaxis, 0, :] - qnts[np.newaxis, 1, :]),
                               np.abs(qnts[np.newaxis, 1, :] - qnts[np.newaxis, 2, :])], axis = 0)
        for k in np.arange(i, j, 1).tolist():
            if hatch_list is not None and k in hatch_list:
                ax1.bar(bar_locs[k % 10] + i / 10, importances_mean_meds[k % 10], width, yerr = qnts[:, k % 10, None],
                        capsize = 2, color = "gray", hatch = "////")
            else:
                ax1.bar(bar_locs[k % 10] + i / 10, importances_mean_meds[k % 10], width, yerr = qnts[:, k % 10, None], capsize = 2, color = "gray")
    for i, j in zip(ranges[:-1], ranges[1:]):
        qnts = np.quantile(nn_imp_mean_ls, [.25, .5, .75], axis = 0)[:, i:j]
        importances_mean_meds = qnts[1, :]
        qnts = np.concatenate([np.abs(qnts[np.newaxis, 0, :] - qnts[np.newaxis, 1, :]),
                               np.abs(qnts[np.newaxis, 1, :] - qnts[np.newaxis, 2, :])], axis = 0)
        for k in np.arange(i, j, 1).tolist():
            if hatch_list is not None and k in hatch_list:
                ax2.bar(bar_locs[k % 10] + i / 10, importances_mean_meds[k % 10], width, yerr = qnts[:, k % 10, None],
                        capsize = 2, color = "gray", hatch = "////")
            else:
                ax2.bar(bar_locs[k % 10] + i / 10, importances_mean_meds[k % 10], width, yerr = qnts[:, k % 10, None], capsize = 2, color = "gray")
    
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(varnames)
    ax1.set_title(knn_title)
    if knn_ylim is not None:
        ax1.set_ylim(knn_ylim)
    ax1.set_ylabel("Permutation Importance")
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels(varnames)
    ax2.set_title(nn_title)
    if nn_ylim is not None:
        ax2.set_ylim(nn_ylim)
    ax2.set_ylabel("Permutation Importance")
    
    return fig

# Make line plot of run times for C2ST studies.
def c2st_time_plot(run_times_ls_ls1,
                   run_times_ls_ls2,
                   sample_size_ls,
                   lab1,
                   lab2,
                   y_lims = None):
    x_ticks = np.arange(len(sample_size_ls) + 1).tolist()[1:]
        
    # Create subplots.
    fig, ax = plt.subplots(constrained_layout = True)
    fig.set_size_inches(5, 5, forward = True)
    
    trans0 = Affine2D().translate(+0.00, 0.0) + ax.transData
    trans1 = Affine2D().translate(+0.00, 0.0) + ax.transData
    
    qnts = [np.quantile(run_times_ls, [.25, .50, .75]) for run_times_ls in run_times_ls_ls1]
    meds = [p[1] for p in qnts]
    err1 = [abs(p[0] - p[1]) for p in qnts]
    err2 = [abs(p[1] - p[2]) for p in qnts]
    p1 = ax.errorbar(x_ticks, meds, yerr = [err1, err2], capsize = 5, fmt = "k-o", transform = trans0, markersize = 3, label = lab1)
    qnts = [np.quantile(run_times_ls, [.25, .50, .75]) for run_times_ls in run_times_ls_ls2]
    meds = [p[1] for p in qnts]
    err1 = [abs(p[0] - p[1]) for p in qnts]
    err2 = [abs(p[1] - p[2]) for p in qnts]
    p2 = ax.errorbar(x_ticks, meds, yerr = [err1, err2], capsize = 5, fmt = "k--^", transform = trans1, markersize = 3, label = lab2)
    handles, labels = ax.get_legend_handles_labels()
    handles = [h[0] for h in handles]
    ax.legend(handles, labels, loc = "upper left", frameon = False, prop = {"size" : 12})
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(sample_size_ls)
    ax.set_xlabel("Test Set Size")
    ax.set_ylabel("Run Time (Seconds)")
    if y_lims is not None:
        ax.set_ylim(y_lims)
    
    return fig