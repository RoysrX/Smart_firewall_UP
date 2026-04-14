"""
Full Experiment Runner — Real CICIDS2017 Friday DDoS Dataset
Paper: Towards Realistic Firewall Anomaly Detection in Label-Scarce Environments

Usage:
    python real_data_run.py --data path/to/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
    python real_data_run.py --data path/to/csv --no-rl
    python real_data_run.py --data path/to/csv --rl-timesteps 200000
"""

import argparse
import warnings
import os
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

FEATURES = [
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Fwd Packet Length Mean", "Bwd Packet Length Mean",
    "Flow Bytes/s", "Flow Packets/s", "Fwd IAT Mean", "Bwd IAT Mean",
    "Packet Length Std", "ACK Flag Count", "PSH Flag Count", "SYN Flag Count",
    "Idle Mean", "Active Mean", "Average Packet Size", "Down/Up Ratio", "Fwd Packets/s",
]

SAMPLE_SIZE = 20_000
RANDOM_STATE = 42


def load_dataset(csv_path: str) -> tuple:
    """Load, clean, sample and split the CICIDS DDoS CSV."""
    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)

    # Strip leading/trailing spaces from column names
    df.columns = df.columns.str.strip()
    print(f"  Raw shape      : {df.shape}")
    print(f"  Label counts   : {df['Label'].value_counts().to_dict()}")

    # Drop NaN / inf
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    print(f"  After cleaning : {df.shape}")

    # Stratified sample — 80% Benign, 20% DDoS (Table 1 paper ratio)
    n_benign = int(SAMPLE_SIZE * 0.80)
    n_attack = int(SAMPLE_SIZE * 0.20)
    benign = df[df["Label"] == "BENIGN"].sample(n=n_benign, random_state=RANDOM_STATE)
    attack = df[df["Label"] == "DDoS"].sample(n=n_attack, random_state=RANDOM_STATE)
    df_s = pd.concat([benign, attack]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"\nStratified sample: {df_s['Label'].value_counts().to_dict()}")

    # Feature selection + scaling
    X = df_s[FEATURES].values.astype(np.float32)
    y = (df_s["Label"] != "BENIGN").astype(int).values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train {X_train.shape}  |  Test {X_test.shape}")
    print(f"Attack prevalence (test): {y_test.mean():.2%}\n")
    return X_train, X_test, y_train, y_test


def main():
    parser = argparse.ArgumentParser(description="Firewall Anomaly Detection — Real DDoS Dataset")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
    parser.add_argument("--no-rl", action="store_true", help="Skip PPO training")
    parser.add_argument("--rl-timesteps", type=int, default=80_000,
                        help="PPO training timesteps (default: 80 000)")
    parser.add_argument("--outdir", type=str, default="outputs_real",
                        help="Output directory for figures (default: outputs_real)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ── Data ────────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = load_dataset(args.data)

    results = {}

    # ── LightGBM ─────────────────────────────────────────────────────────────
    print("=" * 55)
    from supervised_lightgbm import train_lightgbm, evaluate_lightgbm
    lgb = train_lightgbm(X_train, y_train)
    results["LightGBM"] = evaluate_lightgbm(lgb, X_test, y_test)

    # ── Unsupervised ─────────────────────────────────────────────────────────
    from unsupervised_models import run_lof, run_autoencoder, run_isolation_forest
    results["LOF"] = run_lof(X_train, X_test, y_test, contamination=0.20)
    results["Autoencoder"] = run_autoencoder(X_train, y_train, X_test, y_test, epochs=60)
    results["Isolation Forest"] = run_isolation_forest(X_train, X_test, y_test, contamination=0.20)

    # ── PPO ──────────────────────────────────────────────────────────────────
    if not args.no_rl:
        from rl_ppo_agent import train_ppo_agent, evaluate_ppo
        ppo = train_ppo_agent(X_train, y_train, total_timesteps=args.rl_timesteps)
        results["RL (PPO-based)"] = evaluate_ppo(ppo, X_test, y_test)
    else:
        print("\n[--no-rl] Skipping PPO agent.")

    # ── Summary ──────────────────────────────────────────────────────────────
    from visualisation import print_results_table, plot_all_models_bar, plot_pairwise_comparisons
    print("\n📊  RESULTS — Friday DDoS (CICIDS2017)")
    print_results_table(results)

    plot_all_models_bar(results, save_path=os.path.join(args.outdir, "fig3_bar.png"))
    plot_pairwise_comparisons(results, save_path=os.path.join(args.outdir, "fig4_pairwise.png"))

    # Save raw results
    with open(os.path.join(args.outdir, "results.pkl"), "wb") as f:
        pickle.dump(results, f)

    print(f"\n✓ Done — outputs saved to: {args.outdir}/")


if __name__ == "__main__":
    main()
