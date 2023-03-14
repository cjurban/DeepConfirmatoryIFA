#!/usr/bin/env python
#
# Author: Christopher J. Urban
# Affil.: L. L. Thurstone Psychometric Laboratory in the Dept. of Psychology
#         and Neuroscience, UNC-Chapel Hill
# E-mail: cjurban@live.unc.edu
#
# Purpose: Code for classifier two-sample tests.
#
###############################################################################


import torch
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.distributions import Distribution, Normal
from deepirtools import IWAVE
from deepirtools.models import SparseLinear, LinearConstraints, GradedBaseModel, NonGradedBaseModel
from deepirtools.utils import (
    ConvergenceChecker,
    tensor_dataset,
)
from pyro.nn import DenseNN
import numpy as np
from typing import List, Optional
import timeit
import warnings


class OptimizationWarning(UserWarning): # TODO: Move to deepirtools.utils.
    pass


def accuracy(logits: torch.Tensor,
             labels: torch.Tensor,
            ):
    return logits.sigmoid().gt(0.5).eq(labels).float().mean()


class DistributionModel(object):
    
    def __init__(self,
                 distribution: Distribution,
                ):
        self.dist = distribution
            
    def sample(self,
               sample_size: int,
              ):
        return {"obs" : self.dist.sample([sample_size])}


class MultinomialBaselineModel(object):
    
    def __init__(self,
                 data: torch.Tensor,
                ):
        sample_size, n_items = data.shape
        M = int(data.max().item()) + 1
        
        probs = torch.zeros([n_items, M])
        for item in range(n_items):
            temp = data[:, item].unique(return_counts = True)[1].div(sample_size)
            probs[item] = F.pad(temp, (0, M - temp.shape[0]), value = 0.)
        self.probs = probs
        self.sample_size = sample_size
            
    def sample(self,
               sample_size: int,
              ):
        return {"obs" : torch.multinomial(self.probs, self.sample_size, replacement = True).T}
    
    @property
    def num_free_parameters(self):
        return self.probs.ne(0).sum().item()


class C2ST(object):
    
    def __init__(self,
                 input_size:     int,
                 net_sizes:      List[int] = [20],
                 learning_rate:  float = 0.001,
                 device:         str = "cpu",
                 log_interval:   int = 100,
                 verbose:        bool = True,
                 n_intervals:    int = 100,
                ):
        self.device = device
        
        self.net = DenseNN(
            input_size, net_sizes, [1],
            nonlinearity = nn.ELU()
        )
        self.optimizer = Adam(
            [{"params" : self.net.parameters()}],
            lr = learning_rate, amsgrad = True,
        )
        self.loss_function = nn.BCEWithLogitsLoss()
        
        self.global_iter = 0
        self.checker = ConvergenceChecker(log_interval = log_interval, n_intervals = n_intervals)
        self.timerecords = {}
        self.verbose = verbose
        
        self.global_fit_res = {}
        self.importances_res = {}
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.normal_(self.net.layers[-1].weight, mean=0., std=0.001)
        nn.init.normal_(self.net.layers[-1].bias, mean=0., std=0.001)
        
    def global_fit_test(self,
                        model,
                        data_real:       torch.Tensor,
                        delta:           float = 0.,
                        batch_size:      int = 32,
                        max_epochs:      int = 100000,
                        **sample_kwargs,
                       ):
        assert 0 <= delta < 0.5, "delta must be in [0, 0.5)."
        p = 0.5 + delta
        
        sample_size = data_real.shape[0]
        
        if self.verbose:
            print("\nSampling model", end = "\n")
        start = timeit.default_timer()
        data_fake = model.sample(sample_size = sample_size, **sample_kwargs)["obs"]
        stop = timeit.default_timer()
        self.timerecords["sample"] = stop - start
        if self.verbose:
            print("\nModel sampled in ", round(stop - start, 2), " seconds", end = "\n")
        
        if self.verbose:
            print("\nFitting classifier", end = "\n")
            start = timeit.default_timer()
            real_train_idxs = torch.multinomial(torch.ones([sample_size], device=self.device), sample_size // 2)
            fake_train_idxs = torch.multinomial(torch.ones([sample_size], device=self.device), sample_size // 2)
            real_test_idxs = np.setdiff1d(range(sample_size), real_train_idxs)
            fake_test_idxs = np.setdiff1d(range(sample_size), fake_train_idxs)
            data_real_train, data_real_test = data_real[real_train_idxs], data_real[real_test_idxs]
            data_fake_train, data_fake_test = data_fake[fake_train_idxs], data_fake[fake_test_idxs]

            loader_kwargs = {
                "batch_size" : batch_size // 2, "shuffle" : True,
                "pin_memory" : self.device == "cuda", "drop_last" : True,
            }
            real_loader =  torch.utils.data.DataLoader(
                tensor_dataset(data = data_real_train), **loader_kwargs,
            )
            fake_loader =  torch.utils.data.DataLoader(
                tensor_dataset(data = data_fake_train), **loader_kwargs,
            )

            epoch = 0
            while not self.checker.converged:
                self.net.train()

                for batch in zip(real_loader, fake_loader):
                    if not self.checker.converged:
                        self.global_iter += 1 
                        self.net.zero_grad()
                        data = torch.cat((batch[0]["y"], batch[1]["y"]), dim = 0).float()
                        labels = torch.cat((torch.zeros([batch[0]["y"].shape[0], 1], device=self.device),
                                            torch.ones([batch[1]["y"].shape[0], 1], device=self.device)), dim = 0)
                        logits = self.net(data)
                        loss = self.loss_function(logits, labels)

                        if not torch.isnan(loss):
                            loss.backward()
                            self.optimizer.step()
                        else:
                            warnings.warn(("NaN loss obtained, ending fitting. "
                                           "Consider increasing batch size or reducing learning rate."),
                                          OptimizationWarning)
                            self.checker.converged = True
                            break

                        self.checker.check_convergence(epoch, self.global_iter, loss.item())
                    else:
                        break

                epoch += 1
                if epoch == max_epochs and not self.checker.converged:
                    warnings.warn(("Failed to converge within " + str(max_epochs) + " epochs."), OptimizationWarning)
                    break

            data = torch.cat((data_real_test, data_fake_test), dim = 0).float()
            logits = self.net(data)
            labels = torch.cat((torch.zeros([data_real_test.shape[0], 1], device=self.device),
                                torch.ones([data_fake_test.shape[0], 1], device=self.device)), dim = 0)
            acc = accuracy(logits, labels)
            p_val = 1 - Normal(loc = torch.tensor([p]),
                               scale = torch.tensor([p * (1 - p) / data.shape[0]]).sqrt()).cdf(acc)
        stop = timeit.default_timer()
        self.timerecords["fit"] = stop - start
        if self.verbose:
            print("\nFitted classifier in ", round(stop - start, 2), " seconds", end = "\n")
        self.global_fit_res =  {
            "acc" : acc,
            "p_val" : p_val,
            "data_real_train" : data_real_train,
            "data_real_test" : data_real_test,
            "data_fake_train" : data_fake_train,
            "data_fake_test" : data_fake_test,
        }
    
    def permutation_importance(self,
                               n_repeats: int = 10,
                              ):
        try:
            data = torch.cat((self.global_fit_res["data_real_test"],
                              self.global_fit_res["data_fake_test"]), dim = 0).float()
            labels = torch.cat((torch.zeros([self.global_fit_res["data_real_test"].shape[0], 1], device=self.device),
                                torch.ones([self.global_fit_res["data_fake_test"].shape[0], 1], device=self.device)), dim = 0)
            sample_size, n_items = data.shape
            importances = torch.zeros([n_repeats, n_items], device=self.device)
            if self.verbose:
                print("\nComputing permutation importances", end = "\n")
            start = timeit.default_timer()
            for item in range(n_items):
                for perm in range(n_repeats):
                    data_perm = data.clone()
                    data_perm[:, item] = data_perm[torch.randperm(sample_size), item]
                    logits = self.net(data_perm)
                    importances[perm, item] = self.acc - accuracy(logits, labels)
            stop = timeit.default_timer()
            self.timerecords["importances"] = stop - start
            if self.verbose:
                print("\nPermutation importances computed in ", round(stop - start, 2), " seconds", end = "\n")
            self.importances_res = {
                "importances" : importances,
                "importances_mean" : importances.mean(dim = 0),
                "importances_std" : importances.std(dim = 0),
            }   
        except AttributeError:
            return None
                    
    @property
    def acc(self):
        try:
            return self.global_fit_res["acc"]
        except AttributeError:
            return None
        
    @property
    def p_val(self):
        try:
            return self.global_fit_res["p_val"]
        except AttributeError:
            return None
                    
    @property
    def importances(self):
        try:
            return self.importances_res["importances"]
        except AttributeError:
            return None
        
    @property
    def importances_mean(self):
        try:
            return self.importances_res["importances_mean"]
        except AttributeError:
            return None
        
    @property
    def importances_std(self):
        try:
            return self.importances_res["importances_std"]
        except AttributeError:
            return None
        
        
def num_free_parameters(model: IWAVE, # TODO: Add mixed model and spline counts; add to IWAVE.
                       ):
    num_free_params = 0
    
    if model.loadings is not None:
        if isinstance(model.model.decoder._loadings, nn.Linear):
            num_free_params += model.loadings.numel()
        elif isinstance(model.model.decoder._loadings, SparseLinear):
            num_free_params += int(model.model.decoder._loadings.Q.sum().item())
        elif isinstance(model.model.decoder._loadings, LinearConstraints):
            num_free_params += torch.linalg.matrix_rank(model.model.decoder._loadings.A.ne(0).float()).item()
            
    if model.intercepts is not None:
        num_free_params += model.intercepts.isnan().logical_not().sum().item()
        if isinstance(model.model.decoder, GradedBaseModel):
            num_free_params -= model.model.decoder._intercepts.ints_mask.eq(0).sum().item()
        elif isinstance(model.model.decoder, NonGradedBaseModel):
            if model.model.decoder.ints_mask is not None:
                num_free_params -= model.model.decoder.ints_mask.eq(0).sum().item()
                
    if not model.model.use_spline_prior:
        size = model.model.cholesky.size
        lower_diag_size = int(size * (size + 1) / 2)
        num_free_params += lower_diag_size
        num_free_params -= model.model.cholesky.fixed_variances * size
        if model.model.cholesky.model_correlations:
            num_free_params -= model.model.cholesky.uncorrelated_tril_idxs.shape[1]
        else:
            num_free_params -= lower_diag_size - size
        
    return num_free_params
        
    
def c2st_rfi(model_prop,
             model_base,
             c2st_prop: C2ST,
             c2st_base: C2ST,
            ):
    M_prop = num_free_parameters(model_prop)
    M_base = model_base.num_free_parameters
    delta_prop = c2st_prop.acc - 0.5
    delta_base = c2st_base.acc - 0.5
    return 1 - (M_prop / M_base) * (delta_prop / delta_base)


def approx_power(N:     int,
                 eps:   float,
                 delta: float,
                 alpha: float,
                ):
    N, eps, delta, alpha = (
        torch.tensor(val) for val in (N, eps, delta, alpha)
    )
    normal = Normal(loc = torch.zeros(1), scale = torch.ones(1))
    numerator = (
        eps * N.sqrt() - (0.25 - delta.pow(2)).sqrt() * normal.icdf(1 - alpha)
    )
    denominator = (
        (0.25 - delta.pow(2) - 2 * delta * eps - eps.pow(2)).sqrt()
    )
    return normal.cdf(numerator / denominator)
    