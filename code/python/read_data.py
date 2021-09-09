#!/usr/bin/env python
#
# Author: Christopher J. Urban
# Code Author: Christopher J. Urban
# Affil.: L. L. Thurstone Psychometric Laboratory in the
#         Dept. of Psychology and Neuroscience, UNC-Chapel Hill
# E-mail: cjurban@live.unc.edu
#
# Purpose: Some functions for loading data sets.
#
###############################################################################

from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from sklearn.model_selection import train_test_split
    
# Read in a data set from a CSV file.
class csv_dataset(Dataset):
    def __init__(self,
                 csv_file,
                 which_split,
                 test_size = None,
                 val_size = None,
                 transform = None):
        """
        Args:
            csv_file    (string): Path to the CSV file.
            which_split (string): Return the training set, the validation set, or the test set.
                                  "full" = full data set;
                                  "train-only" = split data into train and test set, then return only train samples;
                                  "test-only" = split data into train and test set, then return only test samples;
                                  "train" = split data into train, val., and test sets, then return only train samples;
                                  "test" = split data into train, val., and test sets, then return only test samples;
                                  "val" = split data into train, val., and test sets, then return only val. samples.
            test_size   (int): Proportion of data to include in test set. Must be specified which_split is set to something
                               other than "full".
            val_size    (int): Proportion of data to include in validation set. Must be specified when which_split is set to
                               "train", "test", or "val".
            transform   (Transform): Tranformation of the output samples.
        """
        self.which_split = which_split
        self.transform = transform

        csv_data = pd.read_csv(csv_file, sep = ",")
        
        if self.which_split == "full":
            self.df = csv_data
            
        elif self.which_split == "train-only" or self.which_split == "test-only":
            # Split the data into a training set and a test set.
            csv_train, csv_test = train_test_split(csv_data, train_size = 1 - test_size, test_size = test_size, random_state = 45)
            
            if self.which_split == "train-only":
                self.df = csv_train
            elif self.which_split == "test-only":
                self.df = csv_test
            
        else:
            # Split the data into a training set, a validation set, and a test set.
            csv_train, csv_test = train_test_split(csv_data, train_size = 1 - test_size, test_size = test_size, random_state = 45)
            csv_train, csv_val = train_test_split(csv_train, train_size = 1 - val_size, test_size = val_size, random_state = 50)

            if self.which_split == "train":
                self.df = csv_train
            elif self.which_split == "val":
                self.df = csv_val
            elif self.which_split == "test":
                self.df = csv_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = self.df.iloc[idx, :].to_numpy()

        if self.transform:
            sample = self.transform(sample)

        return sample
    
# Convert a tensor to a data set.
class tensor_dataset(Dataset):
    def __init__(self, tensor):
        """
        Args:
            tensor (Tensor): A Tensor to be converted to a data set.
        """
        self.tensor = tensor

    def __len__(self):
        return self.tensor.shape[0]

    def __getitem__(self, idx):
        sample = self.tensor[idx]

        return sample
    
# Convert Numpy arrays in sample to Tensors.
class to_tensor(object):
    def __call__(self, sample):
        return torch.from_numpy(sample)