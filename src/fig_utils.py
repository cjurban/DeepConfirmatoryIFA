#!/usr/bin/env python
#
# Author: Christopher J. Urban
# Affil.: L. L. Thurstone Psychometric Laboratory in the Dept. of Psychology
#         and Neuroscience, UNC-Chapel Hill
# E-mail: cjurban@live.unc.edu
#
# Purpose: Utility functions for creating figures.
#
###############################################################################

import torch
import matplotlib.pyplot as plt
from pylab import setp
import matplotlib.patches as mpatches
from matplotlib.transforms import Affine2D
from typing import Optional, List, Union
from deepirtools import IWAVE
from .c2st import C2ST
from .mhrm import MHRM


def importances_plt(c2sts:    List[C2ST],
                    x_labels: List[str],
                    ylim:     Optional[List[int]] = None,
                    hatches:  Optional[List[int]] = None,
                   ):
    importances = torch.stack([c.importances_mean for c in c2sts], dim = 0)
    quantiles = importances.quantile(torch.tensor([0.25, 0.5, 0.75]), dim = 0)
    yerr = torch.stack(
        (
            quantiles[1] - quantiles[0],
            quantiles[2] - quantiles[1],
        ), dim = 0
    )
    
    fig = plt.figure()
    fig.set_size_inches(8, 5, forward = True)
    gs = fig.add_gridspec(5, 8)
    gs.tight_layout(fig, rect = [0, 0.03, 1, 0.98])
    ax = fig.add_subplot(gs[0:5, :])
    plt.subplots_adjust(wspace = 1.1, hspace = 1.1)

    x_ticks = torch.arange(len(x_labels))
    width = 1 / 12
    bar_locs = torch.cat([-torch.arange(1, 10, 2).flip((0,)),
                           torch.arange(1, 10, 2)], dim = 0) * 0.5 * width
    
    rngs = torch.arange(0, 51, 10).tolist()
    for i, j in zip(rngs[:-1], rngs[1:]):
        for k in torch.arange(i, j, 1).tolist():
            if hatches is not None and k in hatches:
                hatch = "////"
            else:
                hatch = None
            ax.bar(
                bar_locs[k % 10] + i / 10,
                quantiles[1:2, k:k + 1], width,
                yerr = yerr[:, k:k + 1], capsize = 2,
                color = "gray", hatch = hatch,
            )
    
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Permutation Importance")
    
    return fig


def boxplots(vals:             List[List[torch.Tensor]],
             sample_sizes:     List[int],
             y_label:          str,
             cond_labels:      List[str],
             ylim:             Optional[List[int]] = None,
             legend_loc:       str = "upper right",
             hline_locs:       Optional[List[float]] = None,
             hline_linestyles: Optional[List[str]] = None,
             log_scale:        bool = False,
            ):
    x_ticks = torch.arange(len(sample_sizes)).add(1).tolist()
    
    fig = plt.figure()
    fig.set_size_inches(8, 8, forward = True)
    gs = fig.add_gridspec(4, 4)
    gs.tight_layout(fig, rect = [0, 0.03, 1, 0.98])
    ax = fig.add_subplot(gs[:, :])
    plt.subplots_adjust(wspace = 1.1, hspace = 1.1)
    
    if hline_locs is not None:
        if hline_linestyles is None:
            hline_linestyles = ["--" for _ in range(len(hline_locs))]
        for loc, style in zip(hline_locs, hline_linestyles):
            ax.axhline(
                y = loc, color = "lightgray", linestyle = style,
            )
    
    flierprops = dict(
        marker = "o", markerfacecolor = "black",
        markersize = 2, linestyle = "none",
    )
    
    positions = [
        [0.5 + 2 * i for i in range(len(sample_sizes))],
        [1 + 2 * i for i in range(len(sample_sizes))],
        [1.5 + 2 * i for i in range(len(sample_sizes))],
    ]

    b1 = ax.boxplot(
        vals[0], positions = positions[0],
        widths = 0.45, flierprops = flierprops,
        patch_artist = True,
    )
    for b in b1["medians"]:
        setp(b, color = "black")
    for b in b1["boxes"]:
        b.set(facecolor = "white")
        
    b2 = ax.boxplot(
        vals[1], positions = positions[1],
        widths = 0.45, flierprops = flierprops,
        patch_artist = True,
    )
    for b in b2["medians"]:
        setp(b, color = "black")
    for b in b2["boxes"]:
        b.set(facecolor = "gray")
        
    if len(vals) == 3:
        b3 = ax.boxplot(
            vals[2], positions = positions[2],
            widths = 0.45, flierprops = flierprops,
            patch_artist = True,
        )
        for b in b3["medians"]:
            setp(b, color = "white")
        for b in b3["boxes"]:
            b.set(facecolor = "black")

    ax.set_xticks(
        ([1 + 2 * i for i in range(len(sample_sizes))]
         if len(vals) == 3 else
         [0.75 + 2 * i for i in range(len(sample_sizes))]),
    )
    ax.set_xticklabels(sample_sizes)
    ax.set_xlabel("Sample Size")
    ax.set_ylabel(y_label)
    
    circs = []
    circs.append(mpatches.Rectangle(
        (0.5, 0.5), 1, 0.5, facecolor = "white",
        label = cond_labels[0], edgecolor = "black",
        )
    )
    circs.append(mpatches.Rectangle(
        (0.5, 0.5), 1, 0.5, facecolor = "gray", 
        label = cond_labels[1], edgecolor = "black",
        )
    )
    if len(vals) == 3:
        circs.append(mpatches.Rectangle(
            (0.5, 0.5), 1, 0.5, facecolor = "black",
            label = cond_labels[2], edgecolor = "black",
            )
        )
    ax.legend(
        handles = circs, loc = legend_loc, frameon = False, prop = {"size" : 14},
    )
    if log_scale:
        ax.set_yscale("log")
    if ylim is not None:
        ax.set_ylim(ylim)
    
    return fig


def time_plt(models:       List[List[Union[MHRM, IWAVE]]],
             sample_sizes: List[int],
             cond_labels:  List[str],
             x_trans:      Optional[List[float]] = None,
             fmts:         List[str] = ("k-o", "k:s", "k--^", "k-.+"),
             ylim:         Optional[List[int]] = None,
             legend_loc:   str = "upper right",
             log_scale:    bool = False,
             markersize:   float = 3,
            ):
    x_ticks = torch.arange(len(sample_sizes)).add(1).tolist()
    
    fig = plt.figure()
    fig.set_size_inches(8, 8, forward = True)
    gs = fig.add_gridspec(4, 4)
    gs.tight_layout(fig, rect = [0, 0.03, 1, 0.98])
    ax = fig.add_subplot(gs[:, :])
    plt.subplots_adjust(wspace = 1.1, hspace = 1.1)
    
    if x_trans is None:
        x_trans = [0 for _ in range(len(models))]
    trans = [
        Affine2D().translate(x_trans[i], 0.0) + ax.transData
        for i in range(len(models))
    ]
    
    quantiles = torch.stack([
        torch.stack([
            torch.stack([
                torch.tensor(m3.timerecords["fit"]) for m3 in m2
            ]).quantile(
                torch.tensor([0.25, 0.5, 0.75]), dim = 0,
            ) for m2 in m1
        ]) for m1 in models
    ])
    yerrs = torch.cat(
        (
            quantiles[..., 1:2] - quantiles[..., 0:1],
            quantiles[..., 2:3] - quantiles[..., 1:2],
        ), dim = -1
    )
    
    for i in range(len(models)):
        ax.errorbar(
            x_ticks, quantiles[i, ..., 1],
            yerr = yerrs[i].T, capsize = 5, fmt = fmts[i],
            transform = trans[i], markersize = markersize,
            label = cond_labels[i],
        )

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(sample_sizes)
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Fitting Time (Seconds)")
        
    handles, labels = ax.get_legend_handles_labels()
    handles = [h[0] for h in handles]
    ax.legend(
        handles, labels, loc = legend_loc, frameon = False, prop = {"size" : 14},
    )
    if log_scale:
        ax.set_yscale("log")
    if ylim is not None:
        ax.set_ylim(ylim)
    
    return fig


def rr_acc_plt(rrs:               torch.Tensor,
               accs:              torch.Tensor,
               sample_sizes:      List,
               rr_labels:         Optional[List[str]] = None,
               rr_hline:          bool = False,
               rr_ylim:           Optional[List[float]] = None,
               acc_ylim:          Optional[List[float]] = None,
               rr_legend_loc:     str = "upper left",
               rr_bbox_to_anchor: Optional[tuple] = None,
               acc_line_loc:      float = 0.5,
               rr_trans:          List[float] = None,
               frameon:           bool = False,
              ):
    x_ticks = torch.arange(len(sample_sizes)).add(1).tolist()
    
    fig = plt.figure()
    fig.set_size_inches(12, 5, forward = True)
    gs = fig.add_gridspec(2, 4)
    gs.tight_layout(fig, rect = [0, 0.03, 1, 0.98])
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    ax2 = fig.add_subplot(gs[0:2, 2:])
    plt.subplots_adjust(wspace = 1.1, hspace = 0.05)
    
    linestyles = ("k-o", "k:^")

    if rr_hline:
        ax1.axhline(
            0.05, linestyle = "dashed",
            color = "lightgray",
        )
    if rr_trans is None:
        rr_trans = [0 for _ in range(rrs.size(0))]
    rr_trans = [
        Affine2D().translate(rr_trans[i], 0.0) + ax1.transData
        for i in range(rrs.size(0))
    ]
    for i in range(rrs.size(0)):
        ax1.plot(
            x_ticks, rrs[i], linestyles[i],
            label = (
                rr_labels[i] if rr_labels is not None else rr_labels
            ), transform = rr_trans[i],
        )
    leg = ax1.legend(
        loc = rr_legend_loc, bbox_to_anchor = rr_bbox_to_anchor,
        frameon = frameon, prop = {"size" : 14},
    )
    leg.get_frame().set_edgecolor("black")
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(sample_sizes)
    ax1.set_ylabel("Rejection Rate")
    ax1.set_xlabel("Test Set Size")
    if rr_ylim is not None:
        ax1.set_ylim(rr_ylim)
        
    # Make accuracy line plot.
    ax2.axhline(
        acc_line_loc, linestyle = "dashed",
        color = "lightgray", zorder = 1,
    )
    yerr = torch.stack(
        (
            accs[1] - accs[0],
            accs[2] - accs[1],
        ), dim=0
    )
    p1 = ax2.errorbar(
        x_ticks, accs[1], yerr = yerr, capsize = 5,
        fmt = "k-o", markersize = 3, zorder = 2,
    )
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels(sample_sizes)
    ax2.set_ylabel("Test Set Classification Accuracy")
    ax2.set_xlabel("Test Set Size")
    if acc_ylim is not None:
        ax2.set_ylim(acc_ylim)
    
    return fig