#!/usr/bin/env python
#
# Author: Christopher J. Urban
# Affil.: L. L. Thurstone Psychometric Laboratory in the Dept. of Psychology
#         and Neuroscience, UNC-Chapel Hill
# E-mail: cjurban@live.unc.edu
#
# Purpose: Fit the GRM using MH-RM.
#
###############################################################################

import torch
import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri, packages


numpy2ri.activate()


class MHRM():
    
    def __init__(self,
                 model_specification: str,
                 seed:                int = 0,
                ):
        """Fit Samejima's graded response model using MH-RM."""
        super(MHRM).__init__()
        
        self.model_spec = model_specification
        self.seed = seed
        
        self._loadings = None
        self._intercepts = None
        self._cov = None
        self.timerecords = {}
        self._ll = None
         
    def fit(self,
            data: torch.Tensor,
           ):
        ro.r("rm(list = ls())")
        
        packages.importr("mirt")
        
        Y = ro.r.matrix(
            data.numpy(),
            nrow = data.shape[0],
            ncol = data.shape[1],
        )
        
        ro.r.assign("model_spec", self.model_spec)
        ro.r.assign("seed", self.seed)
        ro.r.assign("Y", Y)
        
        ro.r("""
             set.seed(seed)
             
             model = mirt.model(model_spec)
             out = mirt(
                as.data.frame(Y),
                model = model_spec, 
                itemtype = "graded", 
                method = "MHRM",
                draws = 100
             )
             
             ldgs = do.call(
                 rbind, 
                 lapply(
                     coef(out)[1:length(coef(out)) - 1],
                     function(params) {
                         data.frame(params)[grepl("a", names(data.frame(params)))]
                     }
                 )
             )
             ldgs = as.matrix(ldgs)
             ints = unlist(
                 lapply(
                     coef(out)[1:length(coef(out)) - 1],
                     function(params) {
                         data.frame(params)[grepl("d", names(data.frame(params)))]
                     }
                 )
             )
             cov_mat = matrix(0, nrow = 10, ncol = 10)
             order_mat = which(lower.tri(cov_mat, diag = TRUE), arr.ind = TRUE)
             cov_mat[order_mat] = coef(out)$GroupPars[11:65]
             cov_mat[upper.tri(cov_mat)] = t(cov_mat)[upper.tri(cov_mat)]
             time = t(data.frame(extract.mirt(out, "time")))
             ll = extract.mirt(out, "logLik")
             """
        )
             
        self.loadings = torch.from_numpy(ro.r["ldgs"])
        self.intercepts = torch.from_numpy(ro.r["ints"]).view(self.loadings.shape[0], -1)
        self.cov = torch.from_numpy(ro.r["cov_mat"])
        self.timerecords["MH_draws"] = torch.from_numpy(ro.r("time[3]")).item()
        self.timerecords["Mstep"] = torch.from_numpy(ro.r("time[4]")).item()
        self.ll = torch.from_numpy(ro.r["ll"]).item()