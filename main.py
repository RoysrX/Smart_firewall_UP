"""
Main Experiment Runner
Paper: Towards Realistic Firewall Anomaly Detection in Label-Scarce Environments

Usage
-----
  # With a real CICIDS dataset CSV:
  python main.py --data path/to/combine.csv

  # Demo mode (synthetic data, no CSV required):
  python main.py --demo

  # Skip the PPO agent (faster):
  python main.py --demo --no-rl
"""

from __future__ import annotations

import argparse
import numpy as np
import warnings

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Demo data generator (no real CSV needed)
# ---------------------------------------------------------------------------

def _make_demo_data(n: int = 20_000, n_features: int = 20, seed: int = 42):
    """
    Synthetic dataset that mirrors the class distribution of Table 1:
      Benign 80 %, DoS 11 %, PortScan 6.5 %, Web Attack 2.5 %.
    """
    rng = np.random.default_rng(seed)
    counts = {
        "Benign":     int(n * 0.80),
        "DoS":        int(n * 0.11),
        "PortScan":   int(n * 0.065),
        "Web Attack": int(n * 0.025),
    }

    X_parts, y_parts = [], []
    for cls, cnt in counts.items():
        if cls == "Benign":
            X_parts.append(rng.normal(loc=0.0, scale=1.0, size=(cnt, n_features)))
            y_parts.append(np.zeros(cnt, dtype=int))
        else:
            # Attack traffic: shifted distribution
            X_parts.append(rng.normal(loc=2.5, scale=1.5, size=(cnt, n_features)))
            y_parts.append(np.ones(cnt, dtype=int))

    X = np.vstack(X_parts).astype(np.float32)
    y = np.concatenate(y_parts).astype(int)

    # Shuffle
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Firewall Anomaly Detection – Experiment Runner"
    )
    parser.add_argument("--data", type=str, default=None,
                        help="Path to the CICIDS CSV file (combine.csv)")
    parser.add_argument("--demo", action="store_true",
                        help="Use synthetic demo data instead of a real CSV")
    parser.add_argument("--no-rl", action="store_true",
                        help="Skip PPO training (faster evaluation)")
    parser.add_argument("--rl-timesteps", type=int, default=200_000,
                        help="PPO training timesteps (default: 200 000)")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Fraction of data reserved for testing (default: 0.2)")
    parser.add_argument("--outdir", type=str, default=".",
                        help="Directory to save output figures (default: .)")
    args = parser.parse_args()

    if not args.demo and args.data is None:
        parser.error("Provide --data <path> or use --demo for synthetic data.")

    import os
    os.makedirs(args.outdir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Data preparation
    # ------------------------------------------------------------------
    if args.demo:
        print("[DEMO MODE] Generating synthetic data …")
        X_all, y_all = _make_demo_data()
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=args.test_size,
            stratify=y_all, random_state=42
        )
        y_bin_train = y_train        # already binary in demo
        y_bin_test  = y_test
    else:
        print(f"[REAL DATA] Loading from {args.data} …")
        from data_preprocessing import preprocess
        (X_train, X_test,
         y_train, y_test,
         y_bin_train, y_bin_test,
         scaler, le) = preprocess(args.data, test_size=args.test_size)

    print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")
    print(f"Attack prevalence (test): {y_bin_test.mean():.2%}")

    results: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # 2. Supervised – LightGBM
    # ------------------------------------------------------------------
    from supervised_lightgbm import train_lightgbm, evaluate_lightgbm
    lgb_model = train_lightgbm(X_train, y_bin_train)
    results["LightGBM"] = evaluate_lightgbm(lgb_model, X_test, y_bin_test)

    # ------------------------------------------------------------------
    # 3. Unsupervised
    # ------------------------------------------------------------------
    from unsupervised_models import run_lof, run_autoencoder, run_isolation_forest

    results["LOF"] = run_lof(X_train, X_test, y_bin_test)

    results["Autoencoder"] = run_autoencoder(
        X_train, y_bin_train, X_test, y_bin_test,
        epochs=50
    )

    results["Isolation Forest"] = run_isolation_forest(
        X_train, X_test, y_bin_test
    )

    # ------------------------------------------------------------------
    # 4. Reinforcement Learning – PPO
    # ------------------------------------------------------------------
    if not args.no_rl:
        from rl_ppo_agent import train_ppo_agent, evaluate_ppo
        ppo_model = train_ppo_agent(
            X_train, y_bin_train,
            total_timesteps=args.rl_timesteps,
            alpha=1.0, beta=1.0, gamma_param=2.0
        )
        results["RL (PPO-based)"] = evaluate_ppo(ppo_model, X_test, y_bin_test)
    else:
        print("\n[--no-rl] Skipping PPO agent.")

    # ------------------------------------------------------------------
    # 5. Results summary
    # ------------------------------------------------------------------
    from visualisation import print_results_table, plot_all_models_bar, plot_pairwise_comparisons

    print_results_table(results)

    fig3_path = os.path.join(args.outdir, "fig3_all_models_bar.png")
    fig4_path = os.path.join(args.outdir, "fig4_pairwise.png")

    plot_all_models_bar(results, save_path=fig3_path)
    plot_pairwise_comparisons(results, save_path=fig4_path)

    print("\n✓ Experiment complete.")
    print(f"  Figures saved to: {args.outdir}/")


if __name__ == "__main__":
    main()
