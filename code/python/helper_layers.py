#!/usr/bin/env python
#
# Author: Christopher J. Urban
# Code Author: Christopher J. Urban
# Affil.: L. L. Thurstone Psychometric Laboratory in the
#         Dept. of Psychology and Neuroscience, UNC-Chapel Hill
# E-mail: cjurban@live.unc.edu
#
# Purpose: Helpful layers for building models.
#
###############################################################################

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from code.python.utils import *

# Module for a bias-only layer.
class Bias(nn.Module):
    
    def __init__(self,
                 init_vals):
        """
        Args:
            init_vals (Tensor): Tensor of floats containing initial values.
        """
        super(Bias, self).__init__()
        self.bias = nn.Parameter(init_vals)

    def forward(self, x):
        return self.bias + x
    
# Module for the spherical parameterization of a covariance matrix.
class Spherical(nn.Module):
    
    def __init__(self,
                 dim,
                 correlated_factors,
                 device):
        """
        Args:
            dim                (int): Number of rows and columns.
            correlated_factors (list of int): List of correlated factors.
            device             (str): String specifying whether to run on CPU or GPU.
        """
        super(Spherical, self).__init__()
        
        self.dim = dim
        self.correlated_factors = correlated_factors
        self.device = device
        
        if self.correlated_factors != []:
            n_elts = int((self.dim * (self.dim + 1)) / 2)
            self.theta = nn.Parameter(torch.zeros([n_elts]))
            diag_idxs = torch.from_numpy(np.cumsum(np.arange(1, self.dim + 1)) - 1)
            self.theta.data[diag_idxs] = np.log(np.pi / 2)
            
            # For constraining specific covariances to zero.
            tril_idxs = torch.tril_indices(row = self.dim - 1, col = self.dim - 1, offset = 0)
            uncorrelated_factors = [factor for factor in np.arange(self.dim).tolist() if factor not in self.correlated_factors]
            self.uncorrelated_tril_idxs = tril_idxs[:, sum((tril_idxs[1,:] == factor) + (tril_idxs[0,:] == factor - 1) for
                                                           factor in uncorrelated_factors) > 0]
        
    def weight(self):
        if self.correlated_factors != []:
            tril_idxs = torch.tril_indices(row = self.dim, col = self.dim, offset = 0)
            theta_mat = torch.zeros(self.dim, self.dim).to(self.device)
            theta_mat[tril_idxs[0], tril_idxs[1]] = self.theta

            # Ensure the parameterization is unique.
            exp_theta_mat = torch.zeros(self.dim, self.dim).to(self.device)
            exp_theta_mat[tril_idxs[0], tril_idxs[1]] = self.theta.exp()
            lower_tri_l_mat = (np.pi * exp_theta_mat) / (1 + theta_mat.exp())
            l_mat = exp_theta_mat.diag().diag_embed() + lower_tri_l_mat.tril(diagonal = -1)

            # Constrain variances to one.
            l_mat[:, 0] = torch.ones(l_mat.size(0)).to(self.device)
            
            # Constrain specific covariances to zero.
            l_mat[1:, 1:].data[self.uncorrelated_tril_idxs[0], self.uncorrelated_tril_idxs[1]] = np.pi / 2
        
            return cart2spher(l_mat)
        else:
            return torch.eye(self.dim).to(self.device)

    def forward(self, x):
        return F.linear(x, self.weight())
    
    def cov(self):
        if self.correlated_factors != []:
            weight = self.weight()

            return torch.matmul(weight, weight.t())
        else:
            return torch.eye(self.dim).to(self.device)
    
    def inv_cov(self):
        if self.correlated_factors != []:
            return torch.cholesky_solve(torch.eye(self.dim).to(self.device),
                                        self.weight())
        else:
            return torch.eye(self.dim).to(self.device)
        
# Module for a weight matrix with linear equality constraints.
class LinearConstraints(nn.Module):
    
    def __init__(self,
                 in_features,
                 out_features,
                 A,
                 b = None):
        """
        Args:
            in_features  (int): Size of each input sample.
            out_features (int): Size of each output sample.
            A            (Tensor): Matrix implementing linear constraints.
            b            (Tensor): Vector implementing linear constraints.
        """
        super(LinearConstraints, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.A = A
        self.b = b
        self.theta = nn.Parameter(torch.empty(self.A.shape[0]))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        return init_sparse_xavier_uniform_(self.theta.view(self.in_features, self.out_features).T,
                                           ((self.A.sum(axis = 1) != 0) * 1.).view(self.in_features, self.out_features).T)

    def weight(self):
        return F.linear(self.theta, self.A, self.b).view(self.in_features, self.out_features).T
        
    def forward(self, x):
        return F.linear(x, self.weight(), None)