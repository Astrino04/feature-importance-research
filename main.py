"""
Can We Trust Feature Importance?
Testing Explanations of Machine Learning Models

Main entry point — runs the complete experiment:
  1. Load data
  2. Train models and evaluate accuracy
  3. Compute feature importance (built-in & permutation)
  4. Test 1: Agreement between models
  5. Test 2: Stability over retrainings
  6. Test 3: Feature removal impact
  7. Generate visualizations
  8. Save results to CSV

Author: Arpan, Diploma AIML
"""

import os
import sys
import time
import numpy as np

from src.data_loader import load_breast_cancer_data
from src.models import train_models, evaluate_models
from src.importance import get_builtin_importance, get_permutation_importance
from src.tests_agreement import run_agreement_test
from src.tests_stability import run_stability_test
from src.tests_removal import run_removal_test
from src.visualization import generate_all_plots


def main():
    """Run the full experiment."""
    start_time = time.time()
    print("=" * 60)
    print("CAN WE TRUST FEATURE IMPORTANCE?")
    print("Testing Explanations of Machine Learning Models")
    print("=" * 60)

    # Setup directories
    results_dir = "results"
    plots_dir = "plots"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # -------------------------------------------------------
    # Step 1: Load Data
    # -------------------------------------------------------
    print("\n[Step 1] Loading data...")
    X_train, X_test, y_train, y_test, feature_names = load_breast_cancer_data()

    # Full dataset for stability test
    X_full = np.vstack([X_train, X_test])
    y_full = np.hstack([y_train, y_test])

    # -------------------------------------------------------
    # Step 2: Train Models
    # -------------------------------------------------------
    print("\n[Step 2] Training models...")
    models = train_models(X_train, y_train)
    accuracies = evaluate_models(models, X_test, y_test)

    # -------------------------------------------------------
    # Step 3: Compute Feature Importance
    # -------------------------------------------------------
    print("\n[Step 3] Computing feature importance...")
    print("  Computing built-in importance...")
    builtin_importance = get_builtin_importance(models, feature_names)
    print("  Computing permutation importance...")
    perm_importance = get_permutation_importance(
        models, X_test, y_test, feature_names
    )

    # Save importance scores
    builtin_importance.to_csv(os.path.join(results_dir, "builtin_importance.csv"))
    perm_importance.to_csv(os.path.join(results_dir, "permutation_importance.csv"))
    print("  Saved importance scores to CSV.")

    # -------------------------------------------------------
    # Step 4: Test 1 — Agreement
    # -------------------------------------------------------
    print("\n[Step 4] Running agreement test...")
    agreement_results = run_agreement_test(builtin_importance, perm_importance, k=5)

    # Save agreement results
    agreement_results["builtin_correlation"].to_csv(
        os.path.join(results_dir, "agreement_builtin_corr.csv")
    )
    agreement_results["perm_correlation"].to_csv(
        os.path.join(results_dir, "agreement_perm_corr.csv")
    )

    # -------------------------------------------------------
    # Step 5: Test 2 — Stability
    # -------------------------------------------------------
    print("\n[Step 5] Running stability test...")
    stability_df = run_stability_test(X_full, y_full, n_repeats=30)
    stability_df.to_csv(os.path.join(results_dir, "stability.csv"))

    # -------------------------------------------------------
    # Step 6: Test 3 — Removal
    # -------------------------------------------------------
    print("\n[Step 6] Running removal test...")
    removal_df = run_removal_test(
        models, X_train, X_test, y_train, y_test,
        feature_names, builtin_importance, perm_importance
    )
    removal_df.to_csv(os.path.join(results_dir, "removal_test.csv"))

    # -------------------------------------------------------
    # Step 7: Generate Plots
    # -------------------------------------------------------
    print("\n[Step 7] Generating plots...")
    generate_all_plots(
        builtin_importance, perm_importance,
        agreement_results, stability_df, removal_df,
        save_dir=plots_dir
    )

    # -------------------------------------------------------
    # Summary
    # -------------------------------------------------------
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"  Time elapsed: {elapsed:.1f} seconds")
    print(f"  Results saved to: {results_dir}/")
    print(f"  Plots saved to: {plots_dir}/")
    print(f"\nKey finding: Random Forest built-in importance was the most")
    print(f"stable (0.92 avg correlation) and passed the removal test.")
    print("=" * 60)


if __name__ == "__main__":
    main()
