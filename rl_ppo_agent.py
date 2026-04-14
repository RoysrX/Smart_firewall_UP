"""
Reinforcement Learning: PPO-Based Firewall Policy Adaptation
Paper: Towards Realistic Firewall Anomaly Detection in Label-Scarce Environments
Section 5.3 & 5.4

The task is modelled as a Markov Decision Process (MDP):
  - State  : 20 normalised network-flow features
  - Action : {0 = allow (benign), 1 = block (attack)}
  - Reward : r_t = β·TP_t − α·FP_t − γ·FN_t
             α=1 (FP penalty), β=1 (TP reward), γ=2 (FN penalty)
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from sklearn.metrics import (f1_score, precision_score,
                             recall_score, accuracy_score,
                             classification_report)


# ---------------------------------------------------------------------------
# Custom Gymnasium Environment
# ---------------------------------------------------------------------------

class FirewallEnv(gym.Env):
    """
    Single-step MDP environment that feeds network flow samples to the agent
    one at a time and returns a shaped reward.

    Reward parameters (Section 5.4):
        α = 1  (penalty for false positives)
        β = 1  (reward for true positives)
        γ = 2  (penalty for false negatives – higher because missed attacks
                are more harmful than false alarms in firewalling contexts)
    """

    metadata = {"render_modes": []}

    def __init__(self,
                 X: np.ndarray,
                 y: np.ndarray,
                 alpha: float = 1.0,
                 beta: float = 1.0,
                 gamma: float = 2.0,
                 seed: int = 42):
        super().__init__()
        self.X = X.astype(np.float32)
        self.y = y.astype(int)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self._rng = np.random.default_rng(seed)

        n_features = X.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(n_features,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)   # 0 = allow, 1 = block

        self._idx: int = 0
        self._order: np.ndarray = self._rng.permutation(len(self.X))

    # ------------------------------------------------------------------
    def _next_obs(self) -> np.ndarray:
        return self.X[self._order[self._idx]]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._order = self._rng.permutation(len(self.X))
        self._idx = 0
        return self._next_obs(), {}

    def step(self, action: int):
        true_label = self.y[self._order[self._idx]]

        # Reward shaping  r_t = β·TP − α·FP − γ·FN
        if action == 1 and true_label == 1:     # True Positive
            reward = self.beta
        elif action == 1 and true_label == 0:   # False Positive
            reward = -self.alpha
        elif action == 0 and true_label == 1:   # False Negative
            reward = -self.gamma
        else:                                   # True Negative
            reward = 0.0

        self._idx += 1
        terminated = self._idx >= len(self.X)
        truncated = False

        if terminated:
            obs = self.observation_space.sample()  # dummy final obs
        else:
            obs = self._next_obs()

        return obs, reward, terminated, truncated, {}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_ppo_agent(X_train: np.ndarray,
                    y_train: np.ndarray,
                    total_timesteps: int = 200_000,
                    alpha: float = 1.0,
                    beta: float = 1.0,
                    gamma_param: float = 2.0,
                    seed: int = 42) -> PPO:
    """
    Train a PPO agent using Stable-Baselines3.

    PPO ensures stable policy updates in dynamic network environments
    by clipping the policy gradient to prevent destructive large updates
    (Section 5.3).
    """
    def make_env():
        env = FirewallEnv(X_train, y_train,
                          alpha=alpha, beta=beta, gamma=gamma_param,
                          seed=seed)
        return env

    vec_env = DummyVecEnv([make_env])

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        seed=seed,
    )

    print(f"\nTraining PPO agent for {total_timesteps:,} timesteps …")
    model.learn(total_timesteps=total_timesteps)
    print("PPO training complete.")
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_ppo(model: PPO,
                 X_test: np.ndarray,
                 y_test: np.ndarray) -> dict:
    """
    Run the trained agent over the test set in a single deterministic pass
    and compute classification metrics.
    """
    print("\n=== RL (PPO-based) ===")
    predictions = []
    obs = X_test.astype(np.float32)

    for sample in obs:
        action, _ = model.predict(sample.reshape(1, -1), deterministic=True)
        predictions.append(int(action[0]))

    y_pred = np.array(predictions)
    metrics = {
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
        "accuracy":  accuracy_score(y_test, y_pred),
    }
    print(classification_report(y_test, y_pred,
                                target_names=["Benign", "Attack"],
                                zero_division=0))
    print(f"  [RL-PPO]  P={metrics['precision']:.3f}  "
          f"R={metrics['recall']:.3f}  "
          f"F1={metrics['f1']:.3f}  "
          f"Acc={metrics['accuracy']:.3f}")
    return metrics
