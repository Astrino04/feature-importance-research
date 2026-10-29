"""
Data Loading Module

Loads the Wisconsin Breast Cancer dataset and splits it into
training (70%) and test (30%) sets.
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def load_breast_cancer_data(test_size=0.3, random_state=42):
    """
    Load the Wisconsin Breast Cancer dataset and split into train/test.

    Parameters
    ----------
    test_size : float
        Fraction of data to use for testing (default 0.3).
    random_state : int
        Random seed for reproducibility (default 42).

    Returns
    -------
    X_train : ndarray of shape (n_train, 30)
    X_test : ndarray of shape (n_test, 30)
    y_train : ndarray of shape (n_train,)
    y_test : ndarray of shape (n_test,)
    feature_names : list of str
    """
    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = list(data.feature_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Classes: {np.unique(y)} (0=malignant, 1=benign)")

    return X_train, X_test, y_train, y_test, feature_names


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, feature_names = load_breast_cancer_data()
    print(f"\nFeature names: {feature_names}")
