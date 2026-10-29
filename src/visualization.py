"""
Visualization Module

Generates plots for the research paper:
  1. Feature importance bar charts (built-in vs permutation)
  2. Agreement heatmaps (correlation between models)
  3. Stability comparison chart
  4. Removal test results chart
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def setup_style():
    """Set up matplotlib style for publication-quality plots."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
    })


def plot_feature_importance(importance_df, method_name, save_dir="plots"):
    """
    Plot feature importance for all models side by side.

    Parameters
    ----------
    importance_df : pd.DataFrame
        Importance scores with features as index.
    method_name : str
        'Built-in' or 'Permutation'.
    save_dir : str
        Directory to save the plot.
    """
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))

    for idx, col in enumerate(importance_df.columns):
        ax = axes[idx]
        sorted_imp = importance_df[col].sort_values(ascending=True)
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(sorted_imp)))
        ax.barh(range(len(sorted_imp)), sorted_imp.values, color=colors)
        ax.set_yticks(range(len(sorted_imp)))
        ax.set_yticklabels(sorted_imp.index, fontsize=8)
        ax.set_title(col)
        ax.set_xlabel("Importance Score")

    fig.suptitle(f"{method_name} Feature Importance", fontsize=16, fontweight="bold")
    plt.tight_layout()
    filepath = os.path.join(save_dir, f"importance_{method_name.lower().replace(' ', '_')}.png")
    plt.savefig(filepath, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filepath}")


def plot_agreement_heatmap(corr_df, method_name, save_dir="plots"):
    """
    Plot heatmap of pairwise agreement (Spearman correlation).

    Parameters
    ----------
    corr_df : pd.DataFrame
        Correlation matrix.
    method_name : str
        'Built-in' or 'Permutation'.
    save_dir : str
        Directory to save the plot.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr_df, annot=True, fmt=".2f", cmap="RdYlGn",
        vmin=-1, vmax=1, center=0, ax=ax,
        square=True, linewidths=0.5
    )
    ax.set_title(f"Model Agreement — {method_name} Importance\n(Spearman Rank Correlation)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    filepath = os.path.join(save_dir, f"agreement_{method_name.lower().replace(' ', '_')}.png")
    plt.savefig(filepath, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filepath}")


def plot_stability(stability_df, save_dir="plots"):
    """
    Plot stability comparison as grouped bar chart.

    Parameters
    ----------
    stability_df : pd.DataFrame
        Stability results with models as index.
    save_dir : str
        Directory to save the plot.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(stability_df.index))
    width = 0.35

    bars1 = ax.bar(x - width / 2, stability_df["Permutation Importance"],
                   width, label="Permutation Importance", color="#e74c3c", alpha=0.85)
    bars2 = ax.bar(x + width / 2, stability_df["Built-in Importance"],
                   width, label="Built-in Importance", color="#2ecc71", alpha=0.85)

    ax.set_xlabel("Model")
    ax.set_ylabel("Average Spearman Correlation")
    ax.set_title("Stability of Feature Importance Rankings\n(Over 30 Retrainings)",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(stability_df.index, rotation=15)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="")

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=10)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=10)

    plt.tight_layout()
    filepath = os.path.join(save_dir, "stability.png")
    plt.savefig(filepath, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filepath}")


def plot_removal_test(removal_df, save_dir="plots"):
    """
    Plot removal test results as grouped bar chart.

    Parameters
    ----------
    removal_df : pd.DataFrame
        Removal test results with percentage strings.
    save_dir : str
        Directory to save the plot.
    """
    setup_style()

    # Convert percentage strings to floats
    numeric_df = removal_df.applymap(lambda x: float(x.strip("%")) / 100 if isinstance(x, str) else x)

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(numeric_df.index))
    n_cols = len(numeric_df.columns)
    width = 0.8 / n_cols
    colors = ["#3498db", "#e74c3c", "#f39c12", "#95a5a6"]

    for i, col in enumerate(numeric_df.columns):
        bars = ax.bar(x + i * width - (n_cols - 1) * width / 2,
                      numeric_df[col], width,
                      label=col, color=colors[i], alpha=0.85)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.1%}", xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

    ax.set_xlabel("Model")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Feature Removal Test\n(Accuracy After Removing 10 Features)",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(numeric_df.index, rotation=15)
    ax.set_ylim(0.5, 1.05)
    ax.legend(loc="lower left")

    plt.tight_layout()
    filepath = os.path.join(save_dir, "removal_test.png")
    plt.savefig(filepath, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filepath}")


def generate_all_plots(builtin_importance, perm_importance,
                       agreement_results, stability_df, removal_df,
                       save_dir="plots"):
    """
    Generate all plots for the paper.

    Parameters
    ----------
    builtin_importance : pd.DataFrame
    perm_importance : pd.DataFrame
    agreement_results : dict
    stability_df : pd.DataFrame
    removal_df : pd.DataFrame
    save_dir : str
    """
    os.makedirs(save_dir, exist_ok=True)

    print("\n=== Generating Plots ===")
    plot_feature_importance(builtin_importance, "Built-in", save_dir)
    plot_feature_importance(perm_importance, "Permutation", save_dir)
    plot_agreement_heatmap(agreement_results["builtin_correlation"], "Built-in", save_dir)
    plot_agreement_heatmap(agreement_results["perm_correlation"], "Permutation", save_dir)
    plot_stability(stability_df, save_dir)
    plot_removal_test(removal_df, save_dir)
    print("  All plots generated!")
