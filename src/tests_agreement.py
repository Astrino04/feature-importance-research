"""
Test 1: Agreement Between Models

Measures how much different models agree on which features are important.
Uses Spearman rank correlation and overlap of top-k features.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from itertools import combinations


def compute_agreement_correlation(importance_df):
    """
    Compute pairwise Spearman rank correlation between models' importance rankings.

    Parameters
    ----------
    importance_df : pd.DataFrame
        Importance scores with features as index and models as columns.

    Returns
    -------
    corr_df : pd.DataFrame
        Pairwise correlation matrix.
    """
    model_names = importance_df.columns.tolist()
    n_models = len(model_names)
    corr_matrix = np.ones((n_models, n_models))

    for i, j in combinations(range(n_models), 2):
        rho, _ = spearmanr(
            importance_df.iloc[:, i], importance_df.iloc[:, j]
        )
        corr_matrix[i, j] = rho
        corr_matrix[j, i] = rho

    corr_df = pd.DataFrame(
        corr_matrix, index=model_names, columns=model_names
    )
    return corr_df


def compute_top_k_overlap(importance_df, k=5):
    """
    Compute pairwise overlap in the top-k features between models.

    Parameters
    ----------
    importance_df : pd.DataFrame
        Importance scores.
    k : int
        Number of top features to compare.

    Returns
    -------
    overlap_df : pd.DataFrame
        Pairwise overlap counts.
    """
    model_names = importance_df.columns.tolist()
    n_models = len(model_names)
    overlap_matrix = np.zeros((n_models, n_models), dtype=int)

    top_k = {}
    for col in model_names:
        top_k[col] = set(importance_df[col].nlargest(k).index)

    for i in range(n_models):
        for j in range(n_models):
            overlap_matrix[i, j] = len(
                top_k[model_names[i]] & top_k[model_names[j]]
            )

    overlap_df = pd.DataFrame(
        overlap_matrix, index=model_names, columns=model_names
    )
    return overlap_df


def run_agreement_test(builtin_importance, perm_importance, k=5):
    """
    Run the full agreement test and print results.

    Parameters
    ----------
    builtin_importance : pd.DataFrame
        Built-in importance scores.
    perm_importance : pd.DataFrame
        Permutation importance scores.
    k : int
        Top-k features to compare.

    Returns
    -------
    results : dict
        Contains correlation and overlap DataFrames for both methods.
    """
    print("\n" + "=" * 60)
    print("TEST 1: AGREEMENT BETWEEN MODELS")
    print("=" * 60)

    # Built-in importance agreement
    print("\n--- Built-in Importance ---")
    builtin_corr = compute_agreement_correlation(builtin_importance)
    builtin_overlap = compute_top_k_overlap(builtin_importance, k=k)
    print(f"\nSpearman Rank Correlations:")
    print(builtin_corr.round(2).to_string())
    print(f"\nTop-{k} Feature Overlap:")
    print(builtin_overlap.to_string())

    # Permutation importance agreement
    print("\n--- Permutation Importance ---")
    perm_corr = compute_agreement_correlation(perm_importance)
    perm_overlap = compute_top_k_overlap(perm_importance, k=k)
    print(f"\nSpearman Rank Correlations:")
    print(perm_corr.round(2).to_string())
    print(f"\nTop-{k} Feature Overlap:")
    print(perm_overlap.to_string())

    return {
        "builtin_correlation": builtin_corr,
        "builtin_overlap": builtin_overlap,
        "perm_correlation": perm_corr,
        "perm_overlap": perm_overlap,
    }
