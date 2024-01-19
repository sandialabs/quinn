#!/usr/bin/env python
"""Downloads datasets in custom dataset classes."""

import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import torch
from ucimlrepo import fetch_ucirepo


class UCI_abalone(Dataset):
    """Face Landmarks dataset."""

    def __init__(self):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.air_quality = fetch_ucirepo(id=1)
        self.X = self.air_quality.data.features
        self.y = self.air_quality.data.targets
        mapping_dict = {
            category: idx for idx, category in enumerate(self.X["Sex"].unique())
        }
        self.X["Sex"] = self.X["Sex"].map(mapping_dict)

        def normalize_column(column):
            return (column - column.min()) / (column.max() - column.min())

        self.X = self.X.apply(normalize_column)
        self.X = torch.tensor(self.X.values, dtype=torch.float32)
        self.y = torch.tensor(self.y.values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]
