import numpy as np
import pandas as pd

def load_data():
    test_df = pd.read_csv("path/to/sample_answer.csv", header=None, names=["file", "aod"])
    train_features = np.load("path/to/train_features_root.npy")
    test_features = np.load("path/to/test_features_root.npy")
    train_targets = np.load("path/to/train_labels_root.npy")
    train_groups = np.load("path/to/train_groups_root.npy")
    return train_features, test_features, train_targets, train_groups, test_df
