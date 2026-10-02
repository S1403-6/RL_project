import os
import csv
import numpy as np
import torch
import gymnasium as gym

from stable_baselines3 import PPO

from discriminator import GAILDiscriminator
from gail_env import GailNavEnv


GAIL_REWARD_SCALE = 1.0

MAX_LINEAR_VELOCITY = 4.0
MAX_ANGULAR_VELOCITY = 2.5

DISCRIMINATOR_UPDATES = 5
DISCRIMINATOR_BATCH_SIZE = 256
DISCRIMINATOR_ROLLOUT_SIZE = 2048

TOTAL_TIMESTEPS = 100000
GAIL_ITERATIONS = 10

PPO_ROLLOUT_STEPS = 2048
PPO_BATCH_SIZE = 256
PPO_EPOCHS = 10
PPO_LEARNING_RATE = 3e-4

DATASET_PATH = "demo_dataset.npz"
DISCRIMINATOR_PATH = "gail_ppo_discriminator.pth"
MODEL_PATH = "gail_ppo_model"
METRICS_PATH = "gail_ppo_metrics.csv"

BC_WEIGHT_PATH = "imitation_policy_weights.npz"


class GAILRewardWrapper(gym.Wrapper):

    def __init__(self, env, discriminator=None, use_gail_reward=False):
        super().__init__(env)
        self.discriminator = discriminator
        self.use_gail_reward = use_gail_reward
        self.last_obs = None

    def set_discriminator(self, discriminator, use_gail_reward):
        self.discriminator = discriminator
        self.use_gail_reward = use_gail_reward

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_obs = np.asarray(obs, dtype=np.float32).copy()
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        if self.last_obs is None:
            raise RuntimeError(
                "GAILRewardWrapper.step() called before reset()."
            )

        state_t = self.last_obs.copy()

        gail_reward = 0.0

        if self.use_gail_reward and self.discriminator is not None:
            gail_reward = float(
                self.discriminator.predict_reward(
                    state_t,
                    action
                )
            )

            gail_reward *= GAIL_REWARD_SCALE

        next_obs, env_reward, terminated, truncated, info = \
            self.env.step(action)

        self.last_obs = np.asarray(
            next_obs,
            dtype=np.float32
        ).copy()

        total_reward = float(env_reward) + gail_reward

        info = dict(info)
        info["gail_reward"] = float(gail_reward)
        info["env_reward"] = float(env_reward)

        return (
            next_obs,
            total_reward,
            terminated,
            truncated,
            info
        )

def train_discriminator(
    discriminator,
    expert_states,
    expert_actions,
    agent_states,
    agent_actions
):
    losses = []
    expert_losses = []
    agent_losses = []

    n_expert = len(expert_states)
    n_agent = len(agent_states)

    if n_expert == 0 or n_agent == 0:
        raise ValueError("Expert or agent dataset is empty.")

    for _ in range(DISCRIMINATOR_UPDATES):

        expert_batch_size = min(
            DISCRIMINATOR_BATCH_SIZE,
            n_expert
        )

        agent_batch_size = min(
            DISCRIMINATOR_BATCH_SIZE,
            n_agent
        )

        expert_idx = np.random.choice(
            n_expert,
            size=expert_batch_size,
            replace=False
        )

        agent_idx = np.random.choice(
            n_agent,
            size=agent_batch_size,
            replace=False
        )

        loss = discriminator.train_step(
            expert_states[expert_idx],
            expert_actions[expert_idx],
            agent_states[agent_idx],
            agent_actions[agent_idx]
        )

        if isinstance(loss, tuple):
            total_loss = float(loss[0])
            expert_loss = float(loss[1])
            agent_loss = float(loss[2])
        else:
            total_loss = float(loss)
            expert_loss = 0.0
            agent_loss = 0.0

        losses.append(total_loss)
        expert_losses.append(expert_loss)
        agent_losses.append(agent_loss)

    return (
        float(np.mean(losses)),
        float(np.mean(expert_losses)),
        float(np.mean(agent_losses))
    )


def collect_agent_data(model, env, num_samples):
    states = []
    actions = []

    obs, _ = env.reset()

    for _ in range(num_samples):

        action, _ = model.predict(
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

        states.append(obs.copy())
        actions.append(action.copy())

        next_obs, _, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            obs, _ = env.reset()
        else:
            obs = next_obs

    return (
        np.asarray(states, dtype=np.float32),
        np.asarray(actions, dtype=np.float32)
    )


def load_expert_dataset():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Missing {DATASET_PATH}"
        )

    data = np.load(DATASET_PATH)

    expert_states = data["observations"].astype(
        np.float32
    )

    expert_actions_physical = data["actions"].astype(
        np.float32
    )

    if os.path.exists(BC_WEIGHT_PATH):
        bc_data = np.load(BC_WEIGHT_PATH)

        if "mean" in bc_data and "std" in bc_data:
            mean = bc_data["mean"].astype(np.float32)
            std = (
                bc_data["std"].astype(np.float32)
                + 1e-6
            )

            expert_states = (
                expert_states - mean
            ) / std

    expert_actions = np.zeros_like(
        expert_actions_physical,
        dtype=np.float32
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


def evaluate_policy(model, env, episodes=20):
    env.set_discriminator(None, False)

    base_env = env.unwrapped

    results = []

    for _ in range(episodes):

        obs, _ = env.reset()

        previous_pos = base_env.robot_pos.copy()

        path_length = 0.0
        steps = 0

        goal = False
        collision = False
        wall = False

        final_distance = 0.0

        while True:

            action, _ = model.predict(
                obs,
                deterministic=True
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

            next_obs, _, terminated, truncated, info = env.step(
                action
            )

            current_pos = base_env.robot_pos.copy()

            path_length += np.linalg.norm(
                current_pos - previous_pos
            )

            previous_pos = current_pos
            steps += 1

            final_distance = info["dist"]

            goal |= bool(info["goal_reached"])
            collision |= bool(info["collision"])
            wall |= bool(info["wall_hit"])

            obs = next_obs

            if terminated or truncated:
                break

        results.append(
            {
                "goal": int(goal),
                "collision": int(collision),
                "wall": int(wall),
                "final_distance": final_distance,
                "steps": steps,
                "path_length": path_length
            }
        )

    return {
        "goal_success_rate": 100.0 * np.mean(
            [x["goal"] for x in results]
        ),
        "collision_rate": 100.0 * np.mean(
            [x["collision"] for x in results]
        ),
        "wall_hit_rate": 100.0 * np.mean(
            [x["wall"] for x in results]
        ),
        "average_final_distance": float(
            np.mean(
                [x["final_distance"] for x in results]
            )
        ),
        "average_episode_length": float(
            np.mean(
                [x["steps"] for x in results]
            )
        ),
        "average_path_length": float(
            np.mean(
                [x["path_length"] for x in results]
            )
        )
    }


def save_metrics(history):
    with open(
        METRICS_PATH,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "iteration",
                "goal_success_rate",
                "collision_rate",
                "wall_hit_rate",
                "average_final_distance",
                "average_episode_length",
                "average_path_length"
            ]
        )

        writer.writeheader()
        writer.writerows(history)


def train_gail_ppo():

    print("=" * 75)
    print("              GAIL + PPO TRAINING")
    print("=" * 75)

    expert_states, expert_actions = load_expert_dataset()

    print(
        f"Expert samples : {len(expert_states)}"
    )

    base_env = GailNavEnv(
        render_gui=False,
        normalize_obs=True
    )

    discriminator = GAILDiscriminator(
        state_dim=32,
        action_dim=2,
        hidden_dim=128,
        lr=3e-4
    )

    env = GAILRewardWrapper(
        base_env,
        discriminator=None,
        use_gail_reward=False
    )

    policy_kwargs = dict(
        net_arch=dict(
            pi=[128, 64],
            vf=[128, 64]
        ),
        activation_fn=torch.nn.ReLU
    )

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=PPO_LEARNING_RATE,
        n_steps=PPO_ROLLOUT_STEPS,
        batch_size=PPO_BATCH_SIZE,
        n_epochs=PPO_EPOCHS,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1
    )

    history = []

    steps_per_iteration = (
        TOTAL_TIMESTEPS // GAIL_ITERATIONS
    )

    for iteration in range(GAIL_ITERATIONS):

        print()
        print("=" * 75)
        print(
            f"GAIL ITERATION "
            f"{iteration + 1}/{GAIL_ITERATIONS}"
        )
        print("=" * 75)

        print(
            "[1/4] Collecting PPO agent transitions..."
        )

        env.set_discriminator(
            None,
            False
        )

        agent_states, agent_actions = collect_agent_data(
            model,
            env,
            DISCRIMINATOR_ROLLOUT_SIZE
        )

        print(
            f"      Agent samples: "
            f"{len(agent_states)}"
        )

        print(
            "[2/4] Training discriminator..."
        )

        d_loss, expert_loss, agent_loss = train_discriminator(
            discriminator,
            expert_states,
            expert_actions,
            agent_states,
            agent_actions
        )

        d_expert = discriminator.predict_probability(
            expert_states[:1000],
            expert_actions[:1000]
        )

        d_agent = discriminator.predict_probability(
            agent_states[:1000],
            agent_actions[:1000]
        )

        gail_reward = discriminator.predict_reward(
            agent_states[:1000],
            agent_actions[:1000]
        )

        print(
            f"      D loss       : {d_loss:.4f}"
        )

        print(
            f"      Expert BCE   : {expert_loss:.4f}"
        )

        print(
            f"      Agent BCE    : {agent_loss:.4f}"
        )

        print(
            f"      D(expert)    : "
            f"{np.mean(d_expert):.4f}"
        )

        print(
            f"      D(agent)     : "
            f"{np.mean(d_agent):.4f}"
        )

        print(
            f"      GAIL reward  : "
            f"{np.mean(gail_reward):.4f}"
        )

        print(
            "[3/4] PPO training with GAIL reward..."
        )

        env.set_discriminator(
            discriminator,
            True
        )

        model.learn(
            total_timesteps=steps_per_iteration,
            reset_num_timesteps=False
        )

        print(
            "      PPO update complete."
        )

        print(
            "[4/4] Evaluating navigation..."
        )

        metrics = evaluate_policy(
            model,
            env,
            episodes=7
        )

        row = {
            "iteration": iteration + 1,
            **metrics
        }

        history.append(row)

        save_metrics(history)

        print()
        print(
            f"      Goal success : "
            f"{metrics['goal_success_rate']:.2f}%"
        )

        print(
            f"      Collision    : "
            f"{metrics['collision_rate']:.2f}%"
        )

        print(
            f"      Wall hit     : "
            f"{metrics['wall_hit_rate']:.2f}%"
        )

        print(
            f"      Final dist   : "
            f"{metrics['average_final_distance']:.3f} m"
        )

        print(
            f"      Episode len  : "
            f"{metrics['average_episode_length']:.2f}"
        )

        print(
            f"      Path length  : "
            f"{metrics['average_path_length']:.3f} m"
        )

        model.save(MODEL_PATH)

        torch.save(
            discriminator.state_dict(),
            DISCRIMINATOR_PATH
        )

        print(
            f"      Saved {MODEL_PATH}.zip"
        )

    env.close()

    print()
    print("=" * 75)
    print("              GAIL + PPO COMPLETED")
    print("=" * 75)

    print(
        f"Model          : {MODEL_PATH}.zip"
    )

    print(
        f"Discriminator  : {DISCRIMINATOR_PATH}"
    )

    print(
        f"Metrics        : {METRICS_PATH}"
    )


if __name__ == "__main__":
    train_gail_ppo()
