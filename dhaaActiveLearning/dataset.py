import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

dhaa_datasets = ['EGG-8', 'EGG-9', 'LAR-2', 'LEA-53' 'PAR-15', 'PAR-16', 'PRO-6', 'PRO-7']


class Dataset:
    def __init__(self, name, files_path='datasets'):
        self.name = name
        self.features, self.labels, self.filenames = load_dataset(self.name + '_features.csv',
                                                                  self.name + '_labels.csv',
                                                                  self.name + '_filenames.csv',
                                                                  files_path=files_path)
        self.classes = np.unique(self.labels)

    def get_split(self, test_size=0.2, random_state=1):
        x_train, x_test, y_train, y_test, f_train, f_test = train_test_split(self.features, self.labels, self.filenames,
                                                                             test_size=test_size,
                                                                             stratify=self.labels,
                                                                             random_state=random_state)
        return x_train, y_train.ravel(), x_test, y_test.ravel(), f_train.ravel(), f_test.ravel()

    def get_splits(self, n_splits=5, random_state=1):
        assert 1 <= n_splits <= 10
        splits = []
        if n_splits == 1:
            splits.append(self.get_split())
        else:
            skf = StratifiedKFold(n_splits=n_splits, random_state=random_state)
            for train_index, test_index in skf.split(self.features, self.labels):
                x_train, x_test = self.features[train_index], self.features[test_index]
                y_train, y_test = self.labels[train_index], self.labels[test_index]
                f_train, f_test = self.filenames[train_index], self.filenames[test_index]
                data = x_train, y_train.ravel(), x_test, y_test.ravel(), f_train.ravel(), f_test.ravel()
                splits.append(data)
        return splits

    def __repr__(self):
        return self.name

    @classmethod
    def get_names(cls, files_path='datasets'):
        return np.unique(list(map(lambda x: x[:x.rfind('_')], os.listdir(files_path))))


def load_dataset(features_file, labels_file, filenames_file, files_path='datasets', test_size=None):
    features = pd.read_csv(os.path.join(files_path, features_file), header=None).values
    labels = pd.read_csv(os.path.join(files_path, labels_file), header=None).values
    try:
        filenames = pd.read_csv(os.path.join(files_path, filenames_file), header=None).values
    except FileNotFoundError:
        filenames = np.arange(len(labels))
    if test_size is not None:
        x_train, x_test, y_train, y_test, f_train, f_test = train_test_split(features, labels, filenames,
                                                                             test_size=test_size,
                                                                             stratify=labels,
                                                                             random_state=1)
        return x_train, y_train.ravel(), x_test, y_test.ravel(), f_train.ravel(), f_test.ravel()
    return features, labels.ravel(), filenames.ravel()