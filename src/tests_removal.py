"""
Test 3: Removal of Top Features

Removes the top-10 most important features (by each method) and measures
the impact on accuracy. If features are truly important, removing them
should significantly reduce accuracy compared to removing random features.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from .importance import get_top_features


def evaluate_with_removed_features(models_dict, X_train, X_test, y_train, y_test,
                                    features_to_remove, feature_names):
    """
    Retrain and evaluate models after removing specified features.

    Parameters
    ----------
    models_dict : dict
        Original fitted models (used to determine model types).
    X_train : ndarray
        Training features (full).
    X_test : ndarray
        Test features (full).
    y_train : ndarray
        Training labels.
    y_test : ndarray
        Test labels.
    features_to_remove : list of str
        Feature names to remove.
    feature_names : list of str
        All feature names.

    Returns
    -------
    accuracies : dict
        Model name -> accuracy after removal.
    """
    from .models import train_models

    # Get column indices to keep
    remove_indices = [feature_names.index(f) for f in features_to_remove]
    keep_indices = [i for i in range(len(feature_names)) if i not in remove_indices]

    X_train_reduced = X_train[:, keep_indices]
    X_test_reduced = X_test[:, keep_indices]

    # Retrain models on reduced features
    new_models = train_models(X_train_reduced, y_train)

    accuracies = {}
    for name, model in new_models.items():
        y_pred = model.predict(X_test_reduced)
        acc = accuracy_score(y_test, y_pred)
        accuracies[name] = acc

    return accuracies


def run_removal_test(models, X_train, X_test, y_train, y_test,
                     feature_names, builtin_importance, perm_importance,
                     n_remove=10, random_state=42):
    """
    Run the full removal test.

    Parameters
    ----------
    models : dict
        Fitted models.
    X_train, X_test : ndarray
        Training and test features.
    y_train, y_test : ndarray
        Training and test labels.
    feature_names : list of str
        Feature names.
    builtin_importance : pd.DataFrame
        Built-in importance scores.
    perm_importance : pd.DataFrame
        Permutation importance scores.
    n_remove : int
        Number of top features to remove.
    random_state : int
        Random seed for selecting random features.

    Returns
    -------
    removal_df : pd.DataFrame
        Results of the removal test.
    """
    from .models import evaluate_models

    print("\n" + "=" * 60)
    print("TEST 3: FEATURE REMOVAL")
    print("=" * 60)

    # Baseline accuracies
    print("\n--- Baseline (nothing removed) ---")
    baseline = evaluate_models(models, X_test, y_test)

    model_names = list(models.keys())
    results = []

    for name in model_names:
        row = {"Model": name, "Nothing Removed": f"{baseline[name]:.1%}"}

        # Top 10 by built-in importance
        top_builtin = get_top_features(builtin_importance, name, n=n_remove)
        print(f"\n--- Removing top {n_remove} by built-in importance ({name}) ---")
        print(f"  Features: {top_builtin}")
        acc_builtin = evaluate_with_removed_features(
            models, X_train, X_test, y_train, y_test, top_builtin, feature_names
        )
        row[f"Top {n_remove} Built-in"] = f"{acc_builtin[name]:.1%}"

        # Top 10 by permutation importance
        top_perm = get_top_features(perm_importance, name, n=n_remove)
        print(f"\n--- Removing top {n_remove} by permutation importance ({name}) ---")
        print(f"  Features: {top_perm}")
        acc_perm = evaluate_with_removed_features(
            models, X_train, X_test, y_train, y_test, top_perm, feature_names
        )
        row[f"Top {n_remove} Permutation"] = f"{acc_perm[name]:.1%}"

        # 10 random features
        rng = np.random.RandomState(random_state)
        random_features = rng.choice(feature_names, size=n_remove, replace=False).tolist()
        print(f"\n--- Removing {n_remove} random features ({name}) ---")
        print(f"  Features: {random_features}")
        acc_random = evaluate_with_removed_features(
            models, X_train, X_test, y_train, y_test, random_features, feature_names
        )
        row[f"{n_remove} Random"] = f"{acc_random[name]:.1%}"

        results.append(row)

    removal_df = pd.DataFrame(results).set_index("Model")
    print("\n\nRemoval Test Results:")
    print(removal_df.to_string())

    return removal_df
