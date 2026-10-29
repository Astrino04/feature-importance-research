"""
Feature Importance Module

Computes feature importance using two methods:
  1. Built-in importance (coefficients for LR, feature_importances_ for trees)
  2. Permutation importance
"""

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def get_builtin_importance(models, feature_names):
    """
    Extract built-in feature importance from each model.

    For Logistic Regression, uses the absolute value of coefficients.
    For tree-based models, uses the `feature_importances_` attribute.

    Parameters
    ----------
    models : dict
        Dictionary of fitted models.
    feature_names : list of str
        Feature names.

    Returns
    -------
    importance_df : pd.DataFrame
        DataFrame with features as index and model importances as columns.
    """
    importance = {}

    for name, model in models.items():
        if hasattr(model, "coef_"):
            # Logistic Regression: absolute coefficients
            imp = np.abs(model.coef_[0])
        elif hasattr(model, "feature_importances_"):
            # Tree-based models
            imp = model.feature_importances_
        else:
            raise ValueError(f"Model {name} has no built-in importance attribute.")
        importance[name] = imp

    importance_df = pd.DataFrame(importance, index=feature_names)
    importance_df.index.name = "Feature"
    return importance_df


def get_permutation_importance(models, X_test, y_test, feature_names,
                                n_repeats=10, random_state=42):
    """
    Compute permutation importance for each model.

    Parameters
    ----------
    models : dict
        Dictionary of fitted models.
    X_test : ndarray
        Test features.
    y_test : ndarray
        Test labels.
    feature_names : list of str
        Feature names.
    n_repeats : int
        Number of times to permute each feature.
    random_state : int
        Random seed.

    Returns
    -------
    importance_df : pd.DataFrame
        DataFrame with features as index and model importances as columns.
    """
    importance = {}

    for name, model in models.items():
        result = permutation_importance(
            model, X_test, y_test,
            n_repeats=n_repeats,
            random_state=random_state,
            scoring="accuracy"
        )
        importance[name] = result.importances_mean

    importance_df = pd.DataFrame(importance, index=feature_names)
    importance_df.index.name = "Feature"
    return importance_df


def get_top_features(importance_df, model_name, n=10):
    """
    Get the top-n most important features for a given model.

    Parameters
    ----------
    importance_df : pd.DataFrame
        Importance scores.
    model_name : str
        Name of the model column.
    n : int
        Number of top features to return.

    Returns
    -------
    top_features : list of str
        Names of the top features.
    """
    return importance_df[model_name].nlargest(n).index.tolist()


def get_importance_rankings(importance_df):
    """
    Convert importance scores to rankings (1 = most important).

    Parameters
    ----------
    importance_df : pd.DataFrame
        Importance scores.

    Returns
    -------
    rankings_df : pd.DataFrame
        Rankings for each model.
    """
    rankings = importance_df.rank(ascending=False).astype(int)
    return rankings
