#!/usr/bin/env Rscript
#
# Author: Christopher Urban
# Start date: 3/23/2021
# Last modified: 3/24/2021
#
# Purpose: Conduct MH-RM analyses.
#
###############################################################################

rm(list = ls())

if(!require(vroom)) {
  install.packages("vroom", repos = "http://cran.us.r-project.org")
  library(vroom)
}

if(!require(mirt)) {
  install.packages("mirt", repos = "http://cran.us.r-project.org")
  library(mirt)
}

# Convert tibbles to matrices.
as_matrix = function(x, n = 1){
  if(!tibble::is_tibble(x) ) stop("x must be a tibble")
  y = as.matrix.data.frame(x)
  if (n == 1) {y} else {y[,-seq(from = 1, to = ncol(y), by = n)]}
}

# Simulated data directory.
base_dir = dirname(dirname(getwd()))

sim_cells  = 1:4
n_reps     = 100

for (sim_cell in sim_cells) {
  # Create directory for saving results.
  dir.create(file.path(base_dir, "results/"), showWarnings = FALSE)
  dir.create(file.path(base_dir, "results/simulations/"), showWarnings = FALSE)
  dir.create(file.path(base_dir, "results/simulations/mhrm/"), showWarnings = FALSE)
  dir.create(file.path(base_dir, "results/simulations/mhrm/mhrm/"), showWarnings = FALSE)
  res_dir = file.path(base_dir, paste0("results/simulations/mhrm/mhrm/sim_cell_", toString(sim_cell - 1)))
  dir.create(res_dir, showWarnings = FALSE)
  dir.create(file.path(res_dir, "loadings/"), showWarnings = FALSE)
  dir.create(file.path(res_dir, "intercepts/"), showWarnings = FALSE)
  dir.create(file.path(res_dir, "cor/"), showWarnings = FALSE)
  dir.create(file.path(res_dir, "run_time/"), showWarnings = FALSE)
  dir.create(file.path(res_dir, "ll/"), showWarnings = FALSE)
  
  for (rep in 1:n_reps) {
    # Read and format data.
    dat = as_matrix(vroom(file.path(base_dir, paste0("data/simulations/mhrm/sim_cell_", toString(sim_cell - 1), "/data_", toString(rep - 1), ".gz")),
                          col_names = FALSE),
                    n = 1)
    dat = cbind.data.frame(lapply(seq(from = 5, to = ncol(dat), by = 5),
                                  function(i) {rowSums(sweep(dat[, (i - 4):i], MARGIN = 2, c(1, 2, 3, 4, 5), `*`))})); colnames(dat) = 1:ncol(dat)
    
    # Fit model using MH-RM.
    s = "F1  = 1-10
         F2  = 11-20
         F3  = 21-30
         F4  = 31-40
         F5  = 41-50
         F6  = 51-60
         F7  = 61-70
         F8  = 71-80
         F9  = 81-90
         F10 = 91-100
         COV = F1*F2*F3*F4*F5*F6*F7*F8*F9*F10"
    set.seed(rep)
    model = mirt.model(s)
    mhrm_res = mirt(dat, model = model, itemtype = "graded", method = "MHRM")
    
    # Extract MH-RM results.
    ldgs = as.matrix(do.call(rbind, lapply(coef(mhrm_res)[1:length(coef(mhrm_res)) - 1],
                                           function(parvec) {data.frame(parvec)[grepl("a", names(data.frame(parvec)))]})))
    ints = unlist(lapply(coef(mhrm_res)[1:length(coef(mhrm_res)) - 1],
                         function(parvec) {data.frame(parvec)[grepl("d", names(data.frame(parvec)))]}))
    cor_mat = matrix(0, nrow = 10, ncol = 10)
    order_mat = which(lower.tri(cor_mat, diag = TRUE), arr.ind = TRUE)
    cor_mat[order_mat] = coef(mhrm_res)$GroupPars[11:65]
    cor_mat[upper.tri(cor_mat)] <- t(cor_mat)[upper.tri(cor_mat)]
    time = t(data.frame(extract.mirt(mhrm_res, "time")))
    ll = extract.mirt(mhrm_res, "logLik")
    
    # Save results.
    write.table(ldgs,
                file.path(res_dir, paste0("loadings/rep_", toString(rep - 1), ".txt")),
                row.names = FALSE, col.names = FALSE)
    write.table(t(data.frame(ints)),
                file.path(res_dir, paste0("intercepts/rep_", toString(rep - 1), ".txt")),
               row.names = FALSE, col.names = FALSE)
    write.table(cor_mat,
                file.path(res_dir, paste0("cor/rep_", toString(rep - 1), ".txt")),
                row.names = FALSE, col.names = FALSE)
    write.table(time,
                file.path(res_dir, paste0("run_time/rep_", toString(rep - 1), ".txt")),
                row.names = FALSE)
    write.table(ll,
                file.path(res_dir, paste0("ll/rep_", toString(rep - 1), ".txt")),
                row.names = FALSE, col.names = FALSE)
  
  }
}