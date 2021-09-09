#!/usr/bin/env python
#
# Code Author: Christopher J. Urban
# Affil.: L. L. Thurstone Psychometric Laboratory in the
#         Dept. of Psychology and Neuroscience, UNC-Chapel Hill
# E-mail: cjurban@live.unc.edu
#
# Purpose: Code for conducting classifier two-sample tests.
#
###############################################################################

import numpy as np
from scipy.stats import norm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import svm
from sklearn import ensemble
from sklearn import neighbors
from sklearn import neural_network
from sklearn.metrics import accuracy_score

def c2st(X,
         y,
         clf,
         param_grid = None,
         eps = 0.,
         random_state = None,
         cv = False):
    p = 0.5 + eps
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = random_state)
    if param_grid is not None:
        if cv:
            grid_clf = GridSearchCV(clf,
                        param_grid,
                        scoring = "accuracy",
                        cv = 5)
        else:
            np.random.seed(random_state)
            val_idxs = np.random.choice(len(X_train),
                                        np.int(np.ceil(0.25 * len(X_train))),
                                        replace = False)
            split_idxs = [-1 if idx in val_idxs else 0 for idx in np.arange(len(X_train))]
            ps = PredefinedSplit(test_fold = split_idxs)
            grid_clf = GridSearchCV(clf,
                                    param_grid,
                                    scoring = "accuracy",
                                    cv = ps)
        grid_clf.fit(X_train, y_train)
        y_pred = grid_clf. best_estimator_.fit(X_train, y_train).predict(X_test)
    else:
        y_pred = clf.fit(X_train, y_train).predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    p_val = 1 - norm(loc = p, scale = np.sqrt(p * (1 - p) / X_test.shape[0])).cdf(acc)
    
    if param_grid is not None:
        return {"acc" : acc,
                "p_val" : p_val,
                "grid_clf" : grid_clf,
                "X_train" : X_train,
                "X_test" : X_test,
                "y_train" : y_train,
                "y_test" : y_test}
    else:
        return {"acc" : acc,
                "p_val" : p_val,
                "clf" : clf,
                "X_train" : X_train,
                "X_test" : X_test,
                "y_train" : y_train,
                "y_test" : y_test}
    
# Compute C2ST-RFI.
def c2st_rfi(acc_prop,
             acc_base,
             M_prop,
             M_base,
             g):
    """
    Args:
        acc_prop (float): Proposed model accuracy.
        acc_base (float): Baseline model accuracy.
        M_prop   (int): Number of parameters for proposed model.
        M_base   (int): Number of parameters for baseline model.
        g        (function): Scalar-valued function.
    """
    delta_prop = g(acc_prop - 0.5)
    delta_base = g(acc_base - 0.5)
    
    return 1 - (M_prop / M_base) * (delta_prop / delta_base)

def approx_power(N,
                 eps,
                 delta,
                 alpha): 
    return norm(0, 1).cdf(((eps * np.sqrt(N) - np.sqrt(0.25 - delta**2) * norm.ppf(1 - alpha)) /
                           np.sqrt(0.25 - delta**2 - 2 * delta * eps - eps**2)))