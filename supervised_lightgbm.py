"""
Supervised Baseline: LightGBM
Paper: Towards Realistic Firewall Anomaly Detection in Label-Scarce Environments
Section 5.1
"""

import numpy as np
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score
import lightgbm as lgb


def train_lightgbm(X_train: np.ndarray, y_train: np.ndarray,
                   num_leaves: int = 63,
                   n_estimators: int = 300,
                   learning_rate: float = 0.05,
                   random_state: int = 42) -> lgb.LGBMClassifier:
    """
    Train a LightGBM binary classifier.

    The model minimises binary cross-entropy loss and produces the predicted
    probability  ŷ_i = σ(Σ f_m(x_i))  where f_m are individual decision trees
    and σ is the sigmoid function (Section 5.1).
    """
    model = lgb.LGBMClassifier(
        num_leaves=num_leaves,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        objective="binary",
        class_weight="balanced",
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=random_state,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_lightgbm(model: lgb.LGBMClassifier,
                      X_test: np.ndarray,
                      y_test: np.ndarray) -> dict:
    """Return precision, recall, F1, and accuracy for the test set."""
    y_pred = model.predict(X_test)
    metrics = {
        "precision": precision_score(y_test, y_pred, average="binary", zero_division=0),
        "recall":    recall_score(y_test, y_pred, average="binary", zero_division=0),
        "f1":        f1_score(y_test, y_pred, average="binary", zero_division=0),
        "accuracy":  accuracy_score(y_test, y_pred),
    }
    print("\n=== LightGBM (Supervised Baseline) ===")
    print(classification_report(y_test, y_pred,
                                target_names=["Benign", "Attack"], zero_division=0))
    return metrics


def run_lightgbm_cv(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> dict:
    """
    Five-fold cross-validation to obtain mean ± std metrics
    matching Table 2 of the paper.
    """
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = {"precision": [], "recall": [], "f1": [], "accuracy": []}

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        model = train_lightgbm(X[train_idx], y[train_idx])
        fold_metrics = evaluate_lightgbm(model, X[val_idx], y[val_idx])
        for k, v in fold_metrics.items():
            results[k].append(v)
        print(f"  Fold {fold} F1={fold_metrics['f1']:.4f}")

    summary = {k: {"mean": np.mean(v), "std": np.std(v)} for k, v in results.items()}
    print("\n[LightGBM CV Summary]")
    for k, s in summary.items():
        print(f"  {k}: {s['mean']:.3f} ± {s['std']:.3f}")
    return summary
