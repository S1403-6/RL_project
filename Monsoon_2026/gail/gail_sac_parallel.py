import os
import multiprocessing as mp
import numpy as np
import torch

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnvWrapper
from feature_extractor import CNNFeatureExtractor
from obstacle_manager import ObstacleManager
from discriminator import GAILDiscriminator

from gail_train import (
    GailNavEnv,
    GAILDiscriminator,
    CNNFeatureExtractor,
    ObstacleManager,
    TARGET_POS,
    MAX_LINEAR_VELOCITY,
    MAX_ANGULAR_VELOCITY,
    OBSTACLE_SPEED,
    DISCRIMINATOR_UPDATES,
    DISCRIMINATOR_BATCH_SIZE,
    DISCRIMINATOR_ROLLOUT_SIZE,
    BC_WEIGHT_PATH,
    DATASET_PATH,
    GAIL_REWARD_SCALE,
    SAC_LEARNING_RATE,
    SAC_BUFFER_SIZE,
    SAC_BATCH_SIZE,
    SAC_LEARNING_STARTS,
    SAC_TRAIN_FREQ,
    SAC_GRADIENT_STEPS,
    transfer_bc_weights_to_sac,
    train_discriminator_multiple_steps,
    evaluate_policy,
    save_metrics_history,
)

N_ENVS = 8
TOTAL_TIMESTEPS = 100000
GAIL_EPOCHS = 10
EVAL_EPISODES = 20

SAC_BATCH_SIZE_PARALLEL = 512
SAC_TRAIN_FREQ_PARALLEL = 1
SAC_GRADIENT_STEPS_PARALLEL = 1

MODEL_PATH = "gail_sac_parallel_model"
METRICS_PATH = "gail_metrics_parallel.csv"


def make_env(rank):
    def _init():
        env = GailNavEnv(
            render_gui=False,
            normalize_obs=True
        )
        env.set_discriminator(
            None,
            use_gail_reward=False
        )
        return env
    return _init


class GAILRewardVecWrapper(VecEnvWrapper):
    def __init__(self, venv, discriminator):
        super().__init__(venv)
        self.discriminator = discriminator
        self.last_actions = None

    def reset(self):
        self.last_actions = None
        return self.venv.reset()

    def step_async(self, actions):
        self.last_actions = np.asarray(
            actions,
            dtype=np.float32
        ).copy()
        return self.venv.step_async(actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()

        if self.discriminator is not None and self.last_actions is not None:
            gail_rewards = np.asarray(
                self.discriminator.predict_reward(
                    obs,
                    self.last_actions
                ),
                dtype=np.float32
            ).reshape(-1)

            gail_rewards *= GAIL_REWARD_SCALE
            rewards = rewards.astype(np.float32) + gail_rewards

            for i, info in enumerate(infos):
                info["gail_reward"] = float(gail_rewards[i])

        return obs, rewards, dones, infos


def load_expert_data():
    if not os.path.exists(DATASET_PATH):
        from collect_dataset import collect_demonstrations
        print("[INFO] Expert dataset missing. Collecting demonstrations...")
        collect_demonstrations()

    expert_data = np.load(DATASET_PATH)

    expert_states = expert_data["observations"].astype(np.float32)
    expert_actions_physical = expert_data["actions"].astype(np.float32)

    if os.path.exists(BC_WEIGHT_PATH):
        bc_data = np.load(BC_WEIGHT_PATH)

        mean = bc_data["mean"].astype(np.float32)
        std = bc_data["std"].astype(np.float32) + 1e-6

        expert_states = (
            expert_states - mean
        ) / std

    expert_actions = np.zeros_like(
        expert_actions_physical
    )

    expert_actions[:, 0] = (
        2.0
        * (
            expert_actions_physical[:, 0]
            / MAX_LINEAR_VELOCITY
        )
        - 1.0
    )

    expert_actions[:, 1] = (
        expert_actions_physical[:, 1]
        / MAX_ANGULAR_VELOCITY
    )

    expert_actions = np.clip(
        expert_actions,
        -1.0,
        1.0
    )

    return expert_states, expert_actions


def collect_agent_transitions_parallel(
    sac_agent,
    vec_env,
    num_samples
):
    states = []
    actions = []

    obs = vec_env.reset()
    collected = 0

    while collected < num_samples:
        action, _ = sac_agent.predict(
            obs,
            deterministic=False
        )

        action = np.asarray(
            action,
            dtype=np.float32
        )

        action = np.clip(
            action,
            -1.0,
            1.0
        )

        next_obs, _, _, _ = vec_env.step(action)

        remaining = num_samples - collected
        take = min(N_ENVS, remaining)

        states.append(
            obs[:take].copy()
        )

        actions.append(
            action[:take].copy()
        )

        collected += take
        obs = next_obs

    return (
        np.concatenate(states, axis=0)[:num_samples],
        np.concatenate(actions, axis=0)[:num_samples]
    )


def print_discriminator_stats(
    discriminator,
    expert_states,
    expert_actions,
    agent_states,
    agent_actions
):
    expert_probability = discriminator.predict_probability(
        expert_states[:1000],
        expert_actions[:1000]
    )

    agent_probability = discriminator.predict_probability(
        agent_states[:1000],
        agent_actions[:1000]
    )

    gail_reward = discriminator.predict_reward(
        agent_states[:1000],
        agent_actions[:1000]
    )

    print(
        f"      D(expert)          : "
        f"{float(np.mean(expert_probability)):.4f}"
    )
    print(
        f"      D(agent)           : "
        f"{float(np.mean(agent_probability)):.4f}"
    )
    print(
        f"      GAIL Reward        : "
        f"{float(np.mean(gail_reward)):.4f}"
    )


def main():
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    print("=" * 70)
    print("  PARALLEL GAIL-SAC TRAINING")
    print("=" * 70)
    print(f"  Parallel Environments : {N_ENVS}")
    print(f"  Expert Samples        : loading...")
    print(f"  Total Timesteps       : {TOTAL_TIMESTEPS}")
    print(f"  GAIL Iterations       : {GAIL_EPOCHS}")
    print(f"  SAC Batch Size        : {SAC_BATCH_SIZE_PARALLEL}")
    print(f"  SAC Train Frequency   : {SAC_TRAIN_FREQ_PARALLEL}")
    print(f"  SAC Gradient Steps    : {SAC_GRADIENT_STEPS_PARALLEL}")
    print(f"  Device                : {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("=" * 70)

    expert_states, expert_actions = load_expert_data()

    print(f"  Expert Samples        : {len(expert_states)}")

    discriminator = GAILDiscriminator(
        state_dim=32,
        action_dim=2,
        hidden_dim=128,
        lr=3e-4
    )

    if torch.cuda.is_available() and hasattr(discriminator, "to"):
        try:
            discriminator.to("cuda")
            print("  Discriminator         : CUDA")
        except Exception:
            print("  Discriminator         : CPU")
    else:
        print("  Discriminator         : CPU")

    base_vec_env = SubprocVecEnv(
        [make_env(i) for i in range(N_ENVS)],
        start_method="spawn"
    )

    train_env = GAILRewardVecWrapper(
        base_vec_env,
        discriminator
    )

    policy_kwargs = dict(
        net_arch=dict(
            pi=[128, 64],
            qf=[128, 64]
        ),
        activation_fn=torch.nn.ReLU,
        log_std_init=-2.0
    )

    sac_agent = SAC(
        "MlpPolicy",
        train_env,
        policy_kwargs=policy_kwargs,
        learning_rate=SAC_LEARNING_RATE,
        buffer_size=SAC_BUFFER_SIZE,
        batch_size=SAC_BATCH_SIZE_PARALLEL,
        learning_starts=SAC_LEARNING_STARTS,
        train_freq=SAC_TRAIN_FREQ_PARALLEL,
        gradient_steps=SAC_GRADIENT_STEPS_PARALLEL,
        gamma=0.99,
        tau=0.005,
        ent_coef="auto_0.1",
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=1
    )

    transfer_bc_weights_to_sac(
        sac_agent,
        BC_WEIGHT_PATH
    )

    steps_per_epoch = TOTAL_TIMESTEPS // GAIL_EPOCHS
    metrics_history = []

    try:
        for epoch in range(GAIL_EPOCHS):
            print("\n" + "=" * 70)
            print(
                f"  GAIL Iteration {epoch + 1}/{GAIL_EPOCHS}"
            )
            print("=" * 70)

            print("[1/3] Collecting agent transitions in parallel...")

            train_env.discriminator = None

            agent_states, agent_actions = (
                collect_agent_transitions_parallel(
                    sac_agent,
                    base_vec_env,
                    DISCRIMINATOR_ROLLOUT_SIZE
                )
            )

            print(
                f"      Agent samples: {len(agent_states)}"
            )

            print("[2/3] Training discriminator...")

            d_loss, exp_loss, agt_loss = (
                train_discriminator_multiple_steps(
                    discriminator,
                    expert_states,
                    expert_actions,
                    agent_states,
                    agent_actions,
                    num_updates=DISCRIMINATOR_UPDATES,
                    batch_size=DISCRIMINATOR_BATCH_SIZE
                )
            )

            print(
                f"      Discriminator Loss : {d_loss:.4f}"
            )
            print(
                f"      Expert BCE         : {exp_loss:.4f}"
            )
            print(
                f"      Agent BCE          : {agt_loss:.4f}"
            )

            print_discriminator_stats(
                discriminator,
                expert_states,
                expert_actions,
                agent_states,
                agent_actions
            )

            print("[3/3] Training SAC with GAIL reward...")

            train_env.discriminator = discriminator

            sac_agent.learn(
                total_timesteps=steps_per_epoch,
                reset_num_timesteps=False
            )

            print("      SAC update complete.")

            print(
                f"      Evaluating policy over "
                f"{EVAL_EPISODES} episodes..."
            )

            eval_env = GailNavEnv(
                render_gui=False,
                normalize_obs=True
            )

            metrics = evaluate_policy(
                sac_agent,
                eval_env,
                num_episodes=EVAL_EPISODES
            )

            eval_env.close()

            metrics_row = {
                "iteration": epoch + 1,
                **metrics
            }

            metrics_history.append(metrics_row)

            save_metrics_history(
                metrics_history,
                METRICS_PATH
            )

            print(
                f"      Goal success rate       : "
                f"{metrics['goal_success_rate']:.2f}%"
            )
            print(
                f"      Collision rate          : "
                f"{metrics['collision_rate']:.2f}%"
            )
            print(
                f"      Wall-hit rate           : "
                f"{metrics['wall_hit_rate']:.2f}%"
            )
            print(
                f"      Avg final distance      : "
                f"{metrics['average_final_distance']:.3f} m"
            )
            print(
                f"      Avg episode length      : "
                f"{metrics['average_episode_length']:.2f} steps"
            )
            print(
                f"      Avg path length         : "
                f"{metrics['average_path_length']:.3f} m"
            )

            sac_agent.save(MODEL_PATH)

            torch.save(
                discriminator.state_dict(),
                "gail_discriminator_parallel.pth"
            )

    finally:
        train_env.close()

    print("\n" + "=" * 70)
    print("  PARALLEL GAIL-SAC TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Model : {MODEL_PATH}.zip")
    print(f"  Metrics: {METRICS_PATH}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
