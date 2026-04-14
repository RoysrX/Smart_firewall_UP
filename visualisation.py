"""
Visualisation Module
Paper: Towards Realistic Firewall Anomaly Detection in Label-Scarce Environments
Reproduces Fig. 3 (bar chart) and Fig. 4 (radar + bar pairwise comparisons).
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import math


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
COLORS = {
    "LightGBM":        "#6A0DAD",   # purple
    "LOF":             "#E07B39",   # orange
    "Autoencoder":     "#C9A84C",   # tan/gold
    "Isolation Forest":"#5B8DB8",   # steel blue
    "RL (PPO-based)":  "#5B8DB8",   # same blue used in paper
}

METRICS = ["F1-score", "Precision", "Recall", "Accuracy"]
METRIC_KEYS = ["f1", "precision", "recall", "accuracy"]

# Paper Table 2 values (used as fallback / demo)
PAPER_RESULTS: dict[str, dict] = {
    "LightGBM":        {"f1": 0.982, "precision": 0.967, "recall": 0.995, "accuracy": 0.985},
    "LOF":             {"f1": 0.810, "precision": 0.852, "recall": 0.771, "accuracy": 0.854},
    "Autoencoder":     {"f1": 0.768, "precision": 0.688, "recall": 0.871, "accuracy": 0.941},
    "Isolation Forest":{"f1": 0.742, "precision": 0.615, "recall": 0.939, "accuracy": 0.738},
    "RL (PPO-based)":  {"f1": 0.671, "precision": 0.626, "recall": 0.721, "accuracy": 0.920},
}


# ---------------------------------------------------------------------------
# Fig. 3 – Grouped bar chart for all models
# ---------------------------------------------------------------------------

def plot_all_models_bar(results: dict[str, dict] | None = None,
                        save_path: str = "fig3_all_models_bar.png") -> None:
    """
    Grouped bar chart comparing F1, Precision, Recall, Accuracy across
    all five models (Fig. 3 of the paper).
    """
    if results is None:
        results = PAPER_RESULTS

    models = list(results.keys())
    n_models = len(models)
    n_metrics = len(METRIC_KEYS)
    bar_w = 0.18
    x = np.arange(n_models)

    metric_colors = ["#6A0DAD", "#E07B39", "#C9A84C", "#4A4A4A"]
    metric_labels = ["F1-score", "Precision", "Recall", "Accuracy"]

    fig, ax = plt.subplots(figsize=(12, 5))

    for i, (key, label, color) in enumerate(
            zip(METRIC_KEYS, metric_labels, metric_colors)):
        vals = [results[m][key] for m in models]
        bars = ax.bar(x + i * bar_w - 1.5 * bar_w, vals,
                      width=bar_w, label=label, color=color, alpha=0.88)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=6.5, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_xlabel("Models", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_title("Comparison of Anomaly Detection Models", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved → {save_path}")


# ---------------------------------------------------------------------------
# Radar chart helper
# ---------------------------------------------------------------------------

def _radar_chart(ax: plt.Axes,
                 values_a: list[float], label_a: str, color_a: str,
                 values_b: list[float], label_b: str, color_b: str,
                 categories: list[str]) -> None:
    N = len(categories)
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]

    def _plot(vals, color, label):
        v = list(vals) + [vals[0]]
        ax.plot(angles, v, color=color, linewidth=1.5, label=label)
        ax.fill(angles, v, color=color, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=5)
    ax.yaxis.set_tick_params(labelsize=5)
    _plot(values_a, color_a, label_a)
    _plot(values_b, color_b, label_b)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=6)


# ---------------------------------------------------------------------------
# Fig. 4 – Pairwise radar + bar comparison
# ---------------------------------------------------------------------------

def plot_pairwise_comparisons(results: dict[str, dict] | None = None,
                               save_path: str = "fig4_pairwise.png") -> None:
    """
    Two rows × 4 columns:
      Row 1: radar charts  (LightGBM vs each other model)
      Row 2: bar charts    (LightGBM vs each other model)
    """
    if results is None:
        results = PAPER_RESULTS

    others = [m for m in results if m != "LightGBM"]
    lgb_vals = [results["LightGBM"][k] for k in METRIC_KEYS]

    fig, axes = plt.subplots(2, 4, figsize=(18, 8),
                              subplot_kw={"projection": None})
    # Override projection for radar row
    fig.clear()
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    radar_axes = []
    for col in range(4):
        radar_ax = fig.add_subplot(2, 4, col + 1, projection="polar")
        radar_axes.append(radar_ax)
    bar_axes = [fig.add_subplot(2, 4, col + 5) for col in range(4)]

    lgb_color = COLORS["LightGBM"]

    for col, other in enumerate(others):
        other_vals = [results[other][k] for k in METRIC_KEYS]
        other_color = list(COLORS.values())[col + 1]

        # Radar
        _radar_chart(radar_axes[col],
                     lgb_vals, "LightGBM", lgb_color,
                     other_vals, other, other_color,
                     METRICS)
        radar_axes[col].set_title(f"LightGBM vs\n{other}", fontsize=8,
                                   fontweight="bold", pad=12)

        # Bar
        ax = bar_axes[col]
        x = np.arange(len(METRICS))
        w = 0.35
        ax.bar(x - w / 2, lgb_vals, w, label="LightGBM",
               color=lgb_color, alpha=0.85)
        ax.bar(x + w / 2, other_vals, w, label=other,
               color=other_color, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(["F1", "Prec", "Rec", "Acc"], fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.set_title(f"LightGBM vs\n{other}", fontsize=8, fontweight="bold")
        ax.legend(fontsize=6)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.suptitle("Model Comparisons: Radar (top) & Bar (bottom)", fontsize=12,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {save_path}")


# ---------------------------------------------------------------------------
# Table 2 – pretty console print
# ---------------------------------------------------------------------------

def print_results_table(results: dict[str, dict]) -> None:
    header = f"{'Model':<22} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Accuracy':>10}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for model, m in results.items():
        print(f"{model:<22} {m['precision']:>10.3f} {m['recall']:>8.3f} "
              f"{m['f1']:>8.3f} {m['accuracy']:>10.3f}")
    print("=" * len(header))
