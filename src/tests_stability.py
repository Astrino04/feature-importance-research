"""
Test 2: Stability Over Retrainings

Retrains each model multiple times with different random seeds and measures
how stable the feature importance rankings are (using Spearman correlation).
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import permutation_importance


def _create_model(model_name, random_state):
    """Create a fresh model instance by name."""
    if model_name == "Logistic Regression":
        return LogisticRegression(max_iter=10000, random_state=random_state)
    elif model_name == "Random Forest":
        return RandomForestClassifier(n_estimators=100, random_state=random_state)
    elif model_name == "Gradient Boosting":
        return GradientBoostingClassifier(n_estimators=100, random_state=random_state)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def _get_builtin_imp(model):
    """Extract built-in importance from a fitted model."""
    if hasattr(model, "coef_"):
        return np.abs(model.coef_[0])
    else:
        return model.feature_importances_


def compute_stability(X, y, model_name, n_repeats=30, test_size=0.3):
    """
    Retrain a model n_repeats times and measure ranking stability.

    Parameters
    ----------
    X : ndarray
        Full feature matrix.
    y : ndarray
        Full labels.
    model_name : str
        Name of the model to test.
    n_repeats : int
        Number of retrainings.
    test_size : float
        Test set fraction.

    Returns
    -------
    stability : dict
        Contains 'builtin' and 'permutation' average correlations.
    """
    builtin_rankings = []
    perm_rankings = []

    for i in range(n_repeats):
        # New random split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=i, stratify=y
        )

        # Train model
        model = _create_model(model_name, random_state=i)
        model.fit(X_train, y_train)

        # Built-in importance ranking
        builtin_imp = _get_builtin_imp(model)
        builtin_rankings.append(builtin_imp)

        # Permutation importance ranking
        perm_result = permutation_importance(
            model, X_test, y_test,
            n_repeats=10, random_state=i, scoring="accuracy"
        )
        perm_rankings.append(perm_result.importances_mean)

    # Compute average pairwise Spearman correlation
    builtin_corrs = []
    perm_corrs = []
    for i in range(n_repeats):
        for j in range(i + 1, n_repeats):
            rho_b, _ = spearmanr(builtin_rankings[i], builtin_rankings[j])
            builtin_corrs.append(rho_b)
            rho_p, _ = spearmanr(perm_rankings[i], perm_rankings[j])
            perm_corrs.append(rho_p)

    return {
        "builtin": np.mean(builtin_corrs),
        "permutation": np.mean(perm_corrs),
    }


def run_stability_test(X, y, n_repeats=30):
    """
    Run the stability test for all three models.

    Parameters
    ----------
    X : ndarray
        Full feature matrix (train + test combined).
    y : ndarray
        Full labels.
    n_repeats : int
        Number of retrainings per model.

    Returns
    -------
    stability_df : pd.DataFrame
        Stability results table.
    """
    print("\n" + "=" * 60)
    print("TEST 2: STABILITY OVER RETRAININGS")
    print("=" * 60)
    print(f"  Retraining each model {n_repeats} times...")

    model_names = ["Logistic Regression", "Random Forest", "Gradient Boosting"]
    results = []

    for name in model_names:
        print(f"  Testing {name}...", end=" ", flush=True)
        stability = compute_stability(X, y, name, n_repeats=n_repeats)
        results.append({
            "Model": name,
            "Permutation Importance": round(stability["permutation"], 2),
            "Built-in Importance": round(stability["builtin"], 2),
        })
        print(f"Done (builtin={stability['builtin']:.2f}, "
              f"perm={stability['permutation']:.2f})")

    stability_df = pd.DataFrame(results).set_index("Model")
    print("\nStability Results (Average Spearman Correlation):")
    print(stability_df.to_string())

    return stability_df
