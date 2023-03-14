#!/usr/bin/env python
#
# Author: Christopher J. Urban
# Affil.: L. L. Thurstone Psychometric Laboratory in the Dept. of Psychology
#         and Neuroscience, UNC-Chapel Hill
# E-mail: cjurban@live.unc.edu
#
# Purpose: Utility functions.
#
###############################################################################

import pickle
import torch
import pyro.distributions as pydist
import numpy
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri, packages
from typing import Optional


numpy2ri.activate()


def save_obj(obj, name):
    """
    Save an object.
    https://stackoverflow.com/questions/19201290/how-to-save-a-dictionary-to-a-file
    """
    with open(name + ".pkl", "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

        
def load_obj(name):
    """
    Load an object.
    https://stackoverflow.com/questions/19201290/how-to-save-a-dictionary-to-a-file
    """
    with open(name + ".pkl", "rb") as f:
        return pickle.load(f)
    
    
class BaseFactorModelSimulator():
    
    def __init__(self,
                 cov_mat:    torch.Tensor,
                 mean:       torch.Tensor,
                ):
        """Base class for simulating from a latent factor model."""
        super(BaseFactorModelSimulator, self).__init__()

        self.cov_mat = cov_mat
        self.mean = mean
    
    @torch.no_grad()
    def _scores(self,
                sample_size: Optional[int] = None,
               ):
        x_dist = pydist.MultivariateNormal(loc = self.mean,
                                           covariance_matrix = self.cov_mat)
        if self.mean.shape[0] > 1:
            return x_dist.sample()
        else:
            return x_dist.sample([sample_size]).squeeze(dim = -2)
        
    def sample(self):
        raise NotImplementedError
        
        
class GradedResponseModelSimulator(BaseFactorModelSimulator):
    
    def __init__(self,
                 loadings:   torch.Tensor,
                 intercepts: torch.Tensor,
                 cov_mat:    torch.Tensor,
                 mean:       torch.Tensor,
                ):
        """ Simulate from Samejima's graded response model."""
        super().__init__(cov_mat = cov_mat, mean = mean)
        
        self.loadings = loadings
        self.intercepts = intercepts
        
    @torch.no_grad()    
    def sample(self,
               sample_size:   Optional[int] = None,
               x:             Optional[torch.Tensor] = None,
               return_scores: bool = False,
              ):
        assert(not ((sample_size is None) and (x is None))), "Must specify either sample_size or x."
        
        ro.r("rm(list = ls())")
        packages.importr("mirt")

        ldgs_R = ro.r.matrix(self.loadings.numpy(), nrow = self.loadings.shape[0],
                             ncol = self.loadings.shape[1])
        ints_R = ro.r.matrix(self.intercepts.numpy(), nrow = self.intercepts.shape[0],
                             ncol = self.intercepts.shape[1])
        if x is None:
            x = self._scores(sample_size)
        Theta_R = ro.r.matrix(x.numpy(), nrow = x.shape[0], ncol = x.shape[1])
        ro.r.assign("ldgs", ldgs_R); ro.r.assign("ints", ints_R); ro.r.assign("Theta", Theta_R)

        ro.r("""
                if (dim(ints)[2] > 1) {
                  itemtype = ifelse(is.na(ints[, 2]), "2PL", "graded")
                } else if (dim(ints)[2] == 1) {
                  itemtype = rep("2PL", dim(ints)[1])
                }
                ldgs = matrix(as.vector(t(ldgs)), nrow = dim(ldgs)[1], byrow = TRUE)
                ints = matrix(as.vector(t(ints)), nrow = dim(ints)[1], byrow = TRUE)
                Theta = matrix(as.vector(t(Theta)), nrow = dim(Theta)[1], byrow = TRUE)
                Y = simdata(a = ldgs,
                            d = ints,
                            itemtype = itemtype,
                            Theta = Theta
                           )
             """)

        if return_scores:
            return torch.from_numpy(ro.r["Y"]), torch.from_numpy(ro.r["Theta"])
        return torch.from_numpy(ro.r["Y"])