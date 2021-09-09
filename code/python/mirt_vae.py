#!/usr/bin/env python
#
# Author: Christopher J. Urban
# Code Author: Christopher J. Urban
# Affil.: L. L. Thurstone Psychometric Laboratory in the
#         Dept. of Psychology and Neuroscience, UNC-Chapel Hill
# E-mail: cjurban@live.unc.edu
#
# Purpose: Model and instance class for I-WAVE.
#
###############################################################################

import torch
from torch import nn, optim
import torch.distributions as dist
import torch.nn.functional as F
from scipy.linalg import block_diag
import itertools
from code.python.utils import *
from code.python.helper_layers import *
from code.python.base_class import BaseClass

EPS = 1e-8
    
# Variational autoencoder for MIRT module.
class MIRTVAE(nn.Module):
    
    def __init__(self,
                 input_dim,
                 inference_model_dims,
                 latent_dim,
                 n_cats,
                 Q,
                 A,
                 b,
                 device,
                 correlated_factors):
        """
        Args:
            input_dim             (int): Input vector dimension.
            inference_model_dims  (list of int): Inference model neural network layer dimensions.
            latent_dim            (int): Latent vector dimension.
            n_cats                (list of int):  List containing number of categories for each observed variable.
            Q                     (Tensor): Matrix with binary entries indicating measurement structure.
            A                     (Tensor): Matrix implementing linear constraints.
            b                     (Tensor): Vector implementing linear constraints.
            device                (str): String specifying whether to run on CPU or GPU.
            correlated_factors    (Boolean): Whether or not factors should be correlated.
        """
        super(MIRTVAE, self).__init__()

        self.input_dim = input_dim
        self.inf_dims = inference_model_dims
        self.latent_dim = latent_dim
        self.n_cats = n_cats
        self.Q = Q
        self.A = A
        self.b = b
        self.device = device
        self.correlated_factors = correlated_factors
        
        # Define inference model neural network.
        if self.inf_dims != []:
            inf_list = []
            inf_list.append(nn.Linear(self.input_dim, self.inf_dims[0]))
            inf_list.append(nn.ELU())
            if len(self.inf_dims) > 1:
                for k in range(len(self.inf_dims) - 1):
                    inf_list.append(nn.Linear(self.inf_dims[k], self.inf_dims[k + 1]))
                    inf_list.append(nn.ELU())
            self.inf = nn.Sequential(*inf_list)
            self.mu = nn.Linear(self.inf_dims[len(self.inf_dims) - 1], self.latent_dim)
            self.logstd = nn.Linear(self.inf_dims[len(self.inf_dims) - 1], self.latent_dim)
        else:
            self.mu = nn.Linear(self.input_dim, self.latent_dim)
            self.logstd = nn.Linear(self.input_dim, self.latent_dim)
        
        # Define loadings matrix.
        if self.Q is not None:
            self.loadings = nn.Linear(self.latent_dim, len(self.n_cats), bias = False)
            init_sparse_xavier_uniform_(self.loadings.weight, self.Q)
        elif self.A is not None:
            self.loadings = LinearConstraints(self.latent_dim, len(self.n_cats), self.A, self.b)
        else:
            self.loadings = nn.Linear(self.latent_dim, len(self.n_cats), bias = False)
            nn.init.xavier_uniform_(self.loadings.weight)
        
        # Define intercept vector.
        self.intercepts = Bias(torch.from_numpy(np.hstack([logistic_thresholds(n_cat) for n_cat in n_cats])))
        
        # Define block diagonal matrix.
        ones = [np.ones((n_cat - 1, 1)) for n_cat in n_cats]
        self.D = torch.from_numpy(block_diag(*ones)).to(self.device).float()
            
        # Define Cholesky decomposition of factor covariance matrix.
        self.cholesky = Spherical(self.latent_dim, self.correlated_factors, self.device)

    def encode(self,
               x,
               mc_samples,
               iw_samples):
        if self.inf_dims != []:
            hidden = self.inf(x)
        else:
            hidden = x

        # Expand Tensors for Monte Carlo samples.
        mu = self.mu(hidden).unsqueeze(0).expand(mc_samples, hidden.size(0), self.latent_dim)
        logstd = self.logstd(hidden).unsqueeze(0).expand(mc_samples, hidden.size(0), self.latent_dim)
        
        # Expand Tensors for importance-weighted samples.
        mu = mu.unsqueeze(0).expand(torch.Size([iw_samples]) + mu.shape)
        logstd = logstd.unsqueeze(0).expand(torch.Size([iw_samples]) + logstd.shape)
            
        return mu, logstd.clamp(min = np.log(EPS), max = -np.log(EPS))

    def reparameterize(self,
                       mu,
                       logstd):
        # Impute factor scores.
        qz_x = dist.Normal(mu, logstd.exp())

        return qz_x.rsample()
        
    def decode(self,
               z):
        # Compute cumulative probabilities.
        activations = self.intercepts(F.linear(self.loadings(z), self.D))
        cum_probs = activations.sigmoid()
        
        # Set up subtraction of adjacent cumulative probabilities.
        one_idxs = np.cumsum(self.n_cats) - 1
        zero_idxs = one_idxs - (np.asarray(self.n_cats) - 1)
        upper_probs = torch.ones(cum_probs.shape[:-1] + torch.Size([cum_probs.size(-1) + len(self.n_cats)])).to(self.device)
        lower_probs = torch.zeros(cum_probs.shape[:-1] + torch.Size([cum_probs.size(-1) + len(self.n_cats)])).to(self.device)
        upper_probs[..., torch.from_numpy(np.delete(np.arange(0, upper_probs.size(-1), 1), one_idxs))] = cum_probs
        lower_probs[..., torch.from_numpy(np.delete(np.arange(0, lower_probs.size(-1), 1), zero_idxs))] = cum_probs
        
        return (upper_probs - lower_probs).clamp(min = EPS, max = 1 - EPS)

    def forward(self,
                x,
                mc_samples = 1,
                iw_samples = 1):
        mu, logstd = self.encode(x, mc_samples, iw_samples)
        z = self.reparameterize(mu, logstd)
        
        return self.decode(z), mu, logstd, z
    
# Variational autoencoder for exploratory IFA instance class.
class MIRTVAEClass(BaseClass):
    
    def __init__(self,
                 input_dim,
                 inference_model_dims,
                 latent_dim,
                 n_cats,
                 learning_rate,
                 device,
                 Q = None,
                 A = None,
                 b = None,
                 correlated_factors = False,
                 inf_grad_estimator = "dreg",
                 steps_anneal = 0):
        """
        New args:
            n_cats                (list of int): List containing number of categories for each observed variable.
            Q                     (Tensor): Matrix with binary entries indicating measurement structure.
            correlated_factors    (Boolean): Whether or not factors should be correlated.
            inf_grad_estimator    (str): Inference model gradient estimator.
                                         "iwae" = IWAE, "dreg" = DReG.
        """
        super().__init__(input_dim, inference_model_dims, latent_dim, learning_rate,
                         device, steps_anneal)
        
        self.n_cats = n_cats
        self.Q = Q
        self.A = A
        self.b = b
        if self.Q is not None:
            self.Q = self.Q.to(device)
        if self.A is not None:
            self.A = self.A.to(device)
        if self.b is not None:
            self.b = self.b.to(device)
        self.correlated_factors = correlated_factors
        self.inf_grad_estimator = inf_grad_estimator
        self.model = MIRTVAE(input_dim = self.input_dim,
                             inference_model_dims = self.inf_dims,
                             latent_dim = self.latent_dim,
                             n_cats = self.n_cats,
                             Q = self.Q,
                             A = self.A,
                             b = self.b,
                             device = self.device,
                             correlated_factors = self.correlated_factors).to(self.device)
        self.optimizer = optim.Adam([{"params" : self.model.parameters()}],
                                    lr = self.lr,
                                    amsgrad = True)
        
    # Fit for one epoch.
    def step(self,
             data,
             mc_samples,
             iw_samples):
        if self.model.training:
            self.optimizer.zero_grad()
            
        output = self.model(data, mc_samples, iw_samples)
        loss = self.loss_function(data, *output, mc_samples, iw_samples)

        if self.model.training and not torch.isnan(loss):
            # Backprop and update parameters.
            loss.backward()
            self.optimizer.step()
            
            # Set fixed loadings to zero.
            if self.Q is not None:
                self.model.loadings.weight.data.mul_(self.Q)

        return loss
    
    # Compute loss for one batch.
    def loss_function(self,
                      x,
                      recon_x,
                      mu,
                      logstd,
                      z,
                      mc_samples,
                      iw_samples):
        # Give batch same dimensions as reconstructed batch.
        x = x.expand(recon_x.shape)
        
        # Compute log p(x | z).
        log_px_z = (-x * recon_x.log()).sum(dim = -1, keepdim = True)
        
        # Compute log p(z).
        if self.correlated_factors:
            log_pz = dist.MultivariateNormal(torch.zeros_like(z).to(self.device),
                                             scale_tril = self.model.cholesky.weight()).log_prob(z).unsqueeze(-1)
            
        else:
            log_pz = dist.Normal(torch.zeros_like(z).to(self.device),
                                 torch.ones_like(z).to(self.device)).log_prob(z).sum(-1, keepdim = True)
        
        # Compute log q(z | x).
        qz_x = dist.Normal(mu, logstd.exp())
        
        # Stop gradient through inf. model params. if using DReG estimator.
        if self.model.training and iw_samples > 1 and self.inf_grad_estimator == "dreg":
            qz_x_ = qz_x.__class__(qz_x.loc.detach(), qz_x.scale.detach())
        else:
            qz_x_ = qz_x                    
        log_qz_x = qz_x_.log_prob(z).sum(-1, keepdim = True)
        
        # Compute ELBO with annealed KL divergence.
        anneal_reg = (linear_annealing(0, 1, self.global_iter, self.steps_anneal)
                      if self.model.training else 1)
        elbo = log_px_z + anneal_reg * (log_qz_x - log_pz)
        
        # Compute ELBO.
        if iw_samples == 1:
            elbo = elbo.squeeze(0).mean(0)
            if self.model.training:
                return elbo.mean()
            else:
                return elbo.sum()

        # Compute IW-ELBO.
        elif self.inf_grad_estimator == "iwae":
            elbo *= -1
            iw_elbo = math.log(elbo.size(0)) - elbo.logsumexp(dim = 0)
                    
            if self.model.training:
                return iw_elbo.mean()
            else:
                return iw_elbo.mean(0).sum()

        # Compute IW-ELBO with DReG estimator.
        elif self.inf_grad_estimator == "dreg":
            elbo *= -1
            with torch.no_grad():
                # Compute normalized importance weights.
                w_tilda = (elbo - elbo.logsumexp(dim = 0)).exp()
                
                if z.requires_grad:
                    z.register_hook(lambda grad: w_tilda * grad)
            
            if self.model.training:
                return (-w_tilda * elbo).sum(0).mean()
            else:
                return (-w_tilda * elbo).sum()
        
    # Compute pseudo-BIC.
    def bic(self,
            eval_loader,
            iw_samples = 1):
        # Get size of data set.
        N = len(eval_loader.dataset)
        
        # Switch to IWAE bound.
        old_estimator = self.inf_grad_estimator
        self.inf_grad_estimator = "iwae"
        
        # Approximate marginal log-likelihood.
        ll = self.test(eval_loader,
                       mc_samples = 1,
                       iw_samples = iw_samples)
        
        # Switch back to previous bound.
        self.inf_grad_estimator = old_estimator
        
        # Get number of estimated parameters.
        n_params = self.model.intercepts.bias.data.numel()
        if self.A is not None:
            n_params += self.model.loadings.weight().nonzero().size(0)
        else:
            n_params += self.model.loadings.weight.nonzero().size(0) 
            
        # Compute BIC.
        bic = 2 * ll + np.log(N) * n_params
        
        return bic, -ll, n_params