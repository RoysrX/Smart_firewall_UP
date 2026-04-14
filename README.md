# Firewall Anomaly Detection — Code Implementation

Implementation of all models from the paper:
> **"Towards Realistic Firewall Anomaly Detection in Label-Scarce Environments"**  
> Roy et al., IEM-ICDC 2026

---

## File Structure

```
firewall_anomaly_detection/
├── data_preprocessing.py      # Cleaning, stratified sampling, scaling (generic)
├── supervised_lightgbm.py     # LightGBM supervised baseline (Section 5.1)
├── unsupervised_models.py     # LOF, Autoencoder, Isolation Forest (Section 5.2)
├── rl_ppo_agent.py            # PPO-based RL agent + FirewallEnv (Section 5.3/5.4)
├── visualisation.py           # Fig. 3 & Fig. 4 reproductions
├── main.py                    # Runner with --demo (synthetic) mode
├── real_data_run.py           # Runner for real CICIDS DDoS CSV
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

### With real CICIDS2017 Friday-DDoS dataset
```bash
python real_data_run.py --data Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
```

### Skip PPO (faster)
```bash
python real_data_run.py --data path/to/csv --no-rl
```

### Control PPO training steps
```bash
python real_data_run.py --data path/to/csv --rl-timesteps 200000
```

### Demo mode (no dataset needed)
```bash
python main.py --demo
```

---

## Models

| Model | Type | Paper Section |
|---|---|---|
| LightGBM | Supervised (upper-bound) | 5.1 |
| Local Outlier Factor | Unsupervised | 5.2 |
| Autoencoder | Unsupervised (PyTorch) | 5.2 |
| Isolation Forest | Unsupervised | 5.2 |
| PPO Agent | Reinforcement Learning | 5.3 / 5.4 |

---

## Results on Friday-DDoS Dataset (225 745 flows → 20 000 sampled)

| Model | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| LightGBM | 0.998 | 1.000 | **0.999** | 1.000 |
| LOF | 0.165 | 0.174 | 0.169 | 0.658 |
| Autoencoder | 0.757 | 0.624 | **0.684** | 0.885 |
| Isolation Forest | 0.189 | 0.184 | 0.186 | 0.679 |
| RL (PPO-based) | 0.948 | 0.998 | **0.972** | 0.989 |

---

## RL Reward Parameters (Section 5.4)

```
α = 1  (penalty for false positives)
β = 1  (reward for true positives)
γ = 2  (penalty for false negatives — missed attacks are more costly)
```

---

## Dataset

**CICIDS2017** — Friday Working Hours Afternoon DDoS  
Download: https://www.unb.ca/cic/datasets/ids-2017.html  
File: `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`  
Raw size: 225 745 flows · 79 features · Labels: BENIGN / DDoS
