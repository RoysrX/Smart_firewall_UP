"""
Unsupervised Anomaly Detection Models
Paper: Towards Realistic Firewall Anomaly Detection in Label-Scarce Environments
Section 5.2  –  LOF | Autoencoder | Isolation Forest
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (f1_score, precision_score,
                             recall_score, accuracy_score,
                             classification_report)


# ---------------------------------------------------------------------------
# 1.  Local Outlier Factor (LOF)
# ---------------------------------------------------------------------------

def run_lof(X_train: np.ndarray, X_test: np.ndarray,
            y_test: np.ndarray,
            n_neighbors: int = 20,
            contamination: float = 0.2) -> dict:
    """
    LOF detects anomalies by comparing the local density of a data point
    with that of its neighbours. Points with significantly lower density
    are considered anomalous (Section 5.2).
    """
    print("\n=== Local Outlier Factor ===")
    clf = LocalOutlierFactor(n_neighbors=n_neighbors,
                             contamination=contamination,
                             novelty=True, n_jobs=-1)
    clf.fit(X_train)
    # sklearn LOF: -1 = outlier (attack), +1 = inlier (benign)
    raw = clf.predict(X_test)
    y_pred = (raw == -1).astype(int)

    metrics = _compute_metrics(y_test, y_pred, "LOF")
    return metrics


# ---------------------------------------------------------------------------
# 2.  Autoencoder  (PyTorch)
# ---------------------------------------------------------------------------

class Autoencoder(nn.Module):
    """
    Encoder → latent representation → Decoder.
    Anomaly score: s(x) = ||x - x̂||²   (Section 5.2)
    """

    def __init__(self, input_dim: int, latent_dim: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        x_hat = self.forward(x)
        return ((x - x_hat) ** 2).mean(dim=1)


def train_autoencoder(X_train_benign: np.ndarray,
                      input_dim: int,
                      latent_dim: int = 8,
                      epochs: int = 50,
                      batch_size: int = 256,
                      lr: float = 1e-3,
                      device: str = "cpu") -> Autoencoder:
    """
    Train autoencoder ONLY on benign traffic so it learns normal behaviour.
    Anomalies produce high reconstruction error at inference.
    """
    model = Autoencoder(input_dim, latent_dim).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    tensor = torch.tensor(X_train_benign, dtype=torch.float32)
    loader = DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            loss = criterion(model(batch), batch)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item() * len(batch)
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}/{epochs}  loss={epoch_loss/len(X_train_benign):.5f}")

    return model


def run_autoencoder(X_train: np.ndarray, y_train: np.ndarray,
                    X_test: np.ndarray, y_test: np.ndarray,
                    latent_dim: int = 8,
                    epochs: int = 50,
                    threshold_percentile: float = 95.0,
                    device: str = "cpu") -> dict:
    """
    Train on benign-only subset, then detect anomalies via reconstruction error.
    """
    print("\n=== Autoencoder ===")
    benign_mask = (y_train == 0)
    X_benign = X_train[benign_mask]

    input_dim = X_train.shape[1]
    model = train_autoencoder(X_benign, input_dim,
                              latent_dim=latent_dim,
                              epochs=epochs,
                              device=device)

    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        errors = model.reconstruction_error(X_tensor).cpu().numpy()

    # Determine threshold from training benign errors
    model.eval()
    with torch.no_grad():
        train_tensor = torch.tensor(X_benign, dtype=torch.float32).to(device)
        train_errors = model.reconstruction_error(train_tensor).cpu().numpy()

    threshold = np.percentile(train_errors, threshold_percentile)
    y_pred = (errors > threshold).astype(int)

    metrics = _compute_metrics(y_test, y_pred, "Autoencoder")
    return metrics


# ---------------------------------------------------------------------------
# 3.  Isolation Forest
# ---------------------------------------------------------------------------

def run_isolation_forest(X_train: np.ndarray, X_test: np.ndarray,
                         y_test: np.ndarray,
                         contamination: float = 0.2,
                         n_estimators: int = 100,
                         random_state: int = 42) -> dict:
    """
    Isolation Forest isolates anomalies through random feature partitioning.
    Samples isolated with fewer splits receive higher anomaly scores (Section 5.2).
    """
    print("\n=== Isolation Forest ===")
    clf = IsolationForest(n_estimators=n_estimators,
                          contamination=contamination,
                          random_state=random_state,
                          n_jobs=-1)
    clf.fit(X_train)
    raw = clf.predict(X_test)          # -1 = anomaly, +1 = normal
    y_pred = (raw == -1).astype(int)

    metrics = _compute_metrics(y_test, y_pred, "Isolation Forest")
    return metrics


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                     name: str) -> dict:
    metrics = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "accuracy":  accuracy_score(y_true, y_pred),
    }
    print(classification_report(y_true, y_pred,
                                target_names=["Benign", "Attack"],
                                zero_division=0))
    print(f"  [{name}]  P={metrics['precision']:.3f}  "
          f"R={metrics['recall']:.3f}  "
          f"F1={metrics['f1']:.3f}  "
          f"Acc={metrics['accuracy']:.3f}")
    return metrics
