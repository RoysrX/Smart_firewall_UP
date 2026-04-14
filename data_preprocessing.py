"""
Data Preprocessing Module
Paper: Towards Realistic Firewall Anomaly Detection in Label-Scarce Environments
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


SELECTED_FEATURES = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Mean",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Fwd IAT Mean",
    "Bwd IAT Mean",
    "Packet Length Std",
    "ACK Flag Count",
    "PSH Flag Count",
    "SYN Flag Count",
    "Idle Mean",
    "Active Mean",
    "Average Packet Size",
    "Down/Up Ratio",
    "Fwd Packets/s",
]

LABEL_COLUMN = "Label"
SAMPLE_SIZE = 20_000
RANDOM_STATE = 42


def load_and_clean(csv_path: str) -> pd.DataFrame:
    """Load CSV, drop rows with NaN / inf / -inf values."""
    df = pd.read_csv(csv_path, low_memory=False)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df


def stratified_sample(df: pd.DataFrame, n: int = SAMPLE_SIZE) -> pd.DataFrame:
    """
    Draw a stratified sample of size *n* preserving the class distribution
    described in Table 1 of the paper (Benign 80 %, DoS 11 %, PortScan 6.5 %,
    Web Attack 2.5 %).
    """
    label_col = LABEL_COLUMN
    # Map the verbose CICIDS labels to four canonical classes
    label_map = {}
    for lbl in df[label_col].unique():
        lbl_lower = str(lbl).lower()
        if lbl_lower == "benign":
            label_map[lbl] = "Benign"
        elif "dos" in lbl_lower or "ddos" in lbl_lower:
            label_map[lbl] = "DoS"
        elif "portscan" in lbl_lower or "port scan" in lbl_lower:
            label_map[lbl] = "PortScan"
        elif "web" in lbl_lower:
            label_map[lbl] = "Web Attack"
        else:
            label_map[lbl] = "Benign"          # fallback

    df = df.copy()
    df[label_col] = df[label_col].map(label_map)

    # Target counts from Table 1
    target_counts = {
        "Benign": int(n * 0.80),
        "DoS": int(n * 0.11),
        "PortScan": int(n * 0.065),
        "Web Attack": int(n * 0.025),
    }

    frames = []
    for cls, count in target_counts.items():
        subset = df[df[label_col] == cls]
        if len(subset) == 0:
            print(f"[WARNING] Class '{cls}' not found in dataset – skipping.")
            continue
        sampled = subset.sample(n=min(count, len(subset)),
                                random_state=RANDOM_STATE)
        frames.append(sampled)

    return pd.concat(frames).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def select_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) using only the 20 selected features."""
    available = [f for f in SELECTED_FEATURES if f in df.columns]
    missing = set(SELECTED_FEATURES) - set(available)
    if missing:
        print(f"[WARNING] Missing features: {missing}")
    X = df[available].copy()
    y = df[LABEL_COLUMN].copy()
    return X, y


def encode_labels(y: pd.Series) -> tuple[np.ndarray, LabelEncoder]:
    """Encode string labels to integers. Returns (y_encoded, encoder)."""
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    return y_enc, le


def build_binary_labels(y_enc: np.ndarray, le: LabelEncoder) -> np.ndarray:
    """Convert multi-class labels to binary (0 = Benign, 1 = Attack)."""
    benign_idx = list(le.classes_).index("Benign")
    return (y_enc != benign_idx).astype(int)


def preprocess(csv_path: str, test_size: float = 0.2):
    """
    Full pipeline:
      1. Load & clean
      2. Stratified sample
      3. Feature selection
      4. Standard scaling
      5. Train / test split
    Returns
    -------
    X_train, X_test, y_train, y_test, y_bin_train, y_bin_test, scaler, le
    """
    df = load_and_clean(csv_path)
    df = stratified_sample(df)
    X, y = select_features(df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    y_enc, le = encode_labels(y)
    y_bin = build_binary_labels(y_enc, le)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_enc, test_size=test_size,
        random_state=RANDOM_STATE, stratify=y_enc
    )
    _, _, y_bin_train, y_bin_test = train_test_split(
        X_scaled, y_bin, test_size=test_size,
        random_state=RANDOM_STATE, stratify=y_enc
    )

    print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")
    print(f"Classes: {le.classes_}")
    return X_train, X_test, y_train, y_test, y_bin_train, y_bin_test, scaler, le
