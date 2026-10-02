# GAIL: Generative Adversarial Imitation Learning for Scaled Dynamic Autonomous Navigation

This directory contains the complete **Generative Adversarial Imitation Learning (GAIL)** navigation framework adapted to a **2.0x scaled environment**, implementing the true **BC Pretraining $\rightarrow$ GAIL Discriminator Reward $\rightarrow$ PPO Optimization** pipeline.

---

## 🚀 Key Differences & Domain Scaling (Relative to Version 3)

All positions, velocities, clearances, and boundary dimensions have been scaled up by **2.0x**:

| Parameter | Version 3 Setting | GAIL Scaled Setting |
| :--- | :--- | :--- |
| **Arena Dimensions** | $[-8, 8] \times [-4, 4]$ m | $[-16, 16] \times [-8, 8]$ m |
| **Start Position** | $(-6.0, 0.0)$ | $(-12.0, 0.0)$ |
| **Target / Goal Position** | $(6.0, 0.0)$ | $(12.0, 0.0)$ |
| **Obstacle Speed** | $0.20$ m/s | $0.40$ m/s |
| **Robot Max Speed $v_{\max}$** | $2.0$ m/s | $4.0$ m/s |
| **Goal Threshold** | $0.25$ m | $0.50$ m |
| **Collision Threshold** | $0.32$ m | $0.64$ m |
| **Wall Threshold** | $0.30$ m | $0.60$ m |
| **Obstacle Separation** | $0.65$ m | $1.30$ m |
| **Safety Critical Clearance**| $0.55$ m | $1.10$ m |
| **Safety Safe Distance** | $1.30$ m | $2.60$ m |

---

## 🏗️ Complete Pipeline Architecture

```
                  EXPERT DEMONSTRATIONS
            {(s_E, a_E)} via Lyapunov Controller
                             │
                             ▼
            ┌──────────────────────────────────┐
            │  Behavioral Cloning Pretraining  │
            └────────────────┬─────────────────┘
                             │
                             ▼
                Pretrained Weights (W, b)
                             │
                             ▼
         ┌──────────────────────────────────────┐
         │     BC WEIGHT TRANSFER TO PPO        │
         │  ppo_agent.policy.actor <= BC W, b   │
         └───────────────────┬──────────────────┘
                             │
                             ▼
             ┌───────────────────────────────┐
             │       ENVIRONMENT ROLLOUT     │
             │ (s_π, a_π) collected by agent │
             └───────────────┬───────────────┘
                             │
                             ▼
             ┌───────────────────────────────┐
             │     PYTORCH DISCRIMINATOR     │
             │      Train D_ψ on BCE Loss    │
             └───────────────┬───────────────┘
                             │
                             ▼
                 Discriminator GAIL Reward
              r_GAIL = - log(1 - D(s,a) + 1e-8)
                             │
                             ▼
             ┌───────────────────────────────┐
             │     STABLE BASELINES3 PPO     │
             │   PPO updates actor on r_GAIL │
             └───────────────┬───────────────┘
                             │
                             ▼
                     Saved GAIL Model
```

---

## 📁 File Breakdown

| File | Functionality |
| :--- | :--- |
| **`collect_dataset.py`** | Collects expert demonstration dataset $(s_t, a_t^E)$ across 120 episodes in PyBullet, saving to `demo_dataset.npz`. |
| **`train_imitation.py`** | Behavioral Cloning (BC) pretraining script training neural policy via MSE loss, saving weights & feature stats to `imitation_policy_weights.npz`. |
| **`discriminator.py`** | PyTorch discriminator neural network $D_\psi(s, a)$ classifying expert vs agent state-action pairs and producing $r_{\text{GAIL}} = -\log(1 - D(s,a) + 1e-8)$. |
| **`gail_train.py`** | Main GAIL training pipeline. Transfers BC policy weights directly into the SB3 PPO Actor, feeds Discriminator GAIL rewards into `GailNavEnv.step()`, and optimizes PPO. Saves to `gail_ppo_model.zip`. |
| **`main.py`** | Main visualization & metric script executing GAIL policy inference directly in PyBullet and saving 3-panel evaluation plots to `gail_evaluation_results.png`. |

---

## 🚀 Execution Instructions

```bash
# 1. Collect Expert Demonstrations
python3 gail/collect_dataset.py

# 2. Behavioral Cloning Pretraining
python3 gail/train_imitation.py

# 3. Train GAIL Policy (BC Weight Transfer -> Discriminator Rewards -> SB3 PPO)
python3 gail/gail_train.py

# 4. Run Visualization & Evaluation
python3 gail/main.py
```
