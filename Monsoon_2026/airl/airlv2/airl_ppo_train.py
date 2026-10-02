import os
import csv
import copy

import numpy as np
import pybullet as p
import torch
import torch.nn as nn
import gymnasium as gym

from stable_baselines3 import PPO

from airl_discriminator import AIRLDiscriminator
from airl_env import AIRLRewardWrapper
from airl_env import AirlNavEnv
from expert_controller import ExpertLyapunovController


AIRL_REWARD_SCALE = 1.0
AIRL_REWARD_CLIP = 10.0
AIRL_REWARD_WEIGHT = 1.0

MAX_LINEAR_VELOCITY = 4.0
MAX_ANGULAR_VELOCITY = 2.5

ENV_WEIGHT = 0.25

DISCRIMINATOR_UPDATES = 3
DISCRIMINATOR_LABEL_SMOOTHING = 0.0
DISCRIMINATOR_BATCH_SIZE = 256
DISCRIMINATOR_ROLLOUT_SIZE = 4096
STATE_DIM = 28

TOTAL_TIMESTEPS = 100000
AIRL_ITERATIONS = 10

TARGET_POS = np.array(
    [12.0, 0.0],
    dtype=np.float32
)

PPO_ROLLOUT_STEPS = 2048
PPO_BATCH_SIZE = 256
PPO_EPOCHS = 10
PPO_LEARNING_RATE = 1e-4

DATASET_PATH = "demo_dataset.npz"

DISCRIMINATOR_PATH = \
    "airl_ppo_discriminator.pth"

MODEL_PATH = "airl_ppo_model"
MODEL_ITER_PREFIX = "airl_ppo_model_iter_"

METRICS_PATH = "airl_ppo_metrics.csv"

BC_WEIGHT_PATH = \
    "imitation_policy_weights.npz"


def get_policy_log_probs(
    model,
    states,
    actions
):

    states_tensor, _ = (
        model.policy.obs_to_tensor(
            states
        )
    )

    actions_tensor = torch.as_tensor(
        actions,
        dtype=torch.float32,
        device=model.device
    )

    with torch.no_grad():

        _, log_prob, _ = (
            model.policy.evaluate_actions(
                states_tensor,
                actions_tensor
            )
        )

    return (
        log_prob
        .detach()
        .cpu()
        .numpy()
        .reshape(-1, 1)
        .astype(np.float32)
    )


def collect_agent_data(
    model,
    env,
    num_samples
):

    states = []
    actions = []
    next_states = []
    dones = []
    env_rewards = []

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

        state = obs.copy()

        next_obs, _, terminated, truncated, info = \
            env.step(action)

        env_rewards.append(float(info.get("env_reward", 0.0)))

        done = (
            terminated
            or truncated
        )

        states.append(
            state
        )

        actions.append(
            action.copy()
        )

        next_states.append(
            next_obs.copy()
        )

        dones.append(
            float(done)
        )

        if done:

            obs, _ = env.reset()

        else:

            obs = next_obs

    states = np.asarray(
        states,
        dtype=np.float32
    )

    actions = np.asarray(
        actions,
        dtype=np.float32
    )

    next_states = np.asarray(
        next_states,
        dtype=np.float32
    )

    dones = np.asarray(
        dones,
        dtype=np.float32
    ).reshape(-1, 1)

    env_rewards = np.asarray(env_rewards, dtype=np.float32)

    log_probs = get_policy_log_probs(
        model,
        states,
        actions
    )

    return (
        states,
        actions,
        next_states,
        dones,
        log_probs,
        env_rewards
    )




def _get_nearest_obstacle_distance(base_env):
    if base_env.obs_mgr is None:
        return np.inf

    try:
        obstacle_positions = base_env.obs_mgr.get_positions()
    except AttributeError:
        return np.inf

    if obstacle_positions is None or len(obstacle_positions) == 0:
        return np.inf

    robot_pos = np.asarray(
        base_env.robot_pos,
        dtype=np.float32
    )

    distances = [
        np.linalg.norm(
            robot_pos
            - np.asarray(
                obstacle[:2],
                dtype=np.float32
            )
        )
        for obstacle in obstacle_positions
    ]

    return float(np.min(distances))


def _get_obstacle_positions(base_env):
    if base_env.obs_mgr is not None:
        for name in (
            "get_positions",
            "get_obstacle_positions",
            "get_current_positions"
        ):
            if hasattr(base_env.obs_mgr, name):
                positions = getattr(
                    base_env.obs_mgr,
                    name
                )()

                if positions is not None:
                    return [
                        np.asarray(
                            obstacle,
                            dtype=np.float32
                        ).copy()
                        for obstacle in positions
                    ]

    positions = []

    for obstacle_id in base_env.obstacle_ids:
        position, _ = p.getBasePositionAndOrientation(
            obstacle_id,
            physicsClientId=base_env.physics_client
        )

        positions.append(
            np.asarray(
                position,
                dtype=np.float32
            )
        )

    return positions


def _nearest_obstacle_distance_from_pybullet(base_env):
    positions = _get_obstacle_positions(base_env)

    if len(positions) == 0:
        return np.inf

    robot_xy = np.asarray(
        base_env.robot_pos,
        dtype=np.float32
    )

    return float(
        min(
            np.linalg.norm(
                robot_xy - obstacle[:2]
            )
            for obstacle in positions
        )
    )


def _snapshot_env_state(base_env, wrapper):
    return {
        "pybullet_state": p.saveState(
            physicsClientId=base_env.physics_client
        ),
        "robot_pos": base_env.robot_pos.copy(),
        "robot_yaw": float(base_env.robot_yaw),
        "step_count": int(base_env.step_count),
        "previous_distance": (
            None
            if base_env.previous_distance is None
            else float(base_env.previous_distance)
        ),
        "obs_mgr": copy.deepcopy(
            base_env.obs_mgr
        ),
        "extractor": copy.deepcopy(
            base_env.extractor
        ),
        "last_obs": (
            None
            if wrapper.last_obs is None
            else wrapper.last_obs.copy()
        )
    }


def _restore_env_state(
    base_env,
    wrapper,
    snapshot
):
    p.restoreState(
        snapshot["pybullet_state"],
        physicsClientId=base_env.physics_client
    )

    base_env.robot_pos = (
        snapshot["robot_pos"].copy()
    )

    base_env.robot_yaw = (
        snapshot["robot_yaw"]
    )

    base_env.step_count = (
        snapshot["step_count"]
    )

    base_env.previous_distance = (
        snapshot["previous_distance"]
    )

    base_env.obs_mgr = copy.deepcopy(
        snapshot["obs_mgr"]
    )

    base_env.extractor = copy.deepcopy(
        snapshot["extractor"]
    )

    wrapper.last_obs = (
        None
        if snapshot["last_obs"] is None
        else snapshot["last_obs"].copy()
    )


class LabelSmoothingBCEWithLogitsLoss(nn.Module):
    def __init__(self, smoothing=0.10):
        super().__init__()
        self.smoothing = float(smoothing)

    def forward(self, logits, targets):
        targets = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        return nn.functional.binary_cross_entropy_with_logits(logits, targets)


def run_airl_action_diagnostic(
    model,
    env,
    discriminator,
    max_search_steps=600,
    obstacle_threshold=1.75
):
    print()
    print("-" * 75)
    print("      AIRL EXPERT-ACTION DIAGNOSTIC")
    print("-" * 75)

    env.set_discriminator(None, False)

    base_env = env.unwrapped
    obs, _ = env.reset()

    near_obstacle = False
    nearest_distance = np.inf

    for _ in range(max_search_steps):

        nearest_distance = _nearest_obstacle_distance_from_pybullet(
            base_env
        )

        if nearest_distance <= obstacle_threshold:
            near_obstacle = True
            break

        action, _ = model.predict(
            obs,
            deterministic=True
        )

        action = np.clip(
            np.asarray(action, dtype=np.float32),
            -1.0,
            1.0
        )

        obs, _, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            obs, _ = env.reset()

    if not near_obstacle:
        print("      Could not find a near-obstacle state.")
        print("      No training parameters were changed.")
        print("-" * 75)
        return

    state = np.asarray(
        env.last_obs,
        dtype=np.float32
    ).copy()

    print(
        f"      Nearest obstacle : {nearest_distance:.3f} m"
    )

    print(
        f"      Robot position   : "
        f"[{base_env.robot_pos[0]:.3f}, "
        f"{base_env.robot_pos[1]:.3f}]"
    )

    print(
        f"      Robot yaw        : "
        f"{base_env.robot_yaw:.3f} rad"
    )


    expert_controller = ExpertLyapunovController(
        Kp=4.0,
        Kd=1.5,
        Kb=8.0,
        K_wall=8.0,
        K_theta=2.5,
        max_v=MAX_LINEAR_VELOCITY,
        max_omega=MAX_ANGULAR_VELOCITY
    )

    obstacle_positions = _get_obstacle_positions(base_env)

    try:
        obstacle_velocities = base_env.obs_mgr.get_velocities()
    except (AttributeError, TypeError):
        obstacle_velocities = None

    robot_velocity = np.asarray(
        getattr(
            base_env,
            "robot_vel",
            np.zeros(2, dtype=np.float32)
        ),
        dtype=np.float32
    )

    expert_v, expert_omega, expert_dist = (
        expert_controller.compute_action(
            base_env.robot_pos,
            robot_velocity,
            base_env.robot_yaw,
            base_env.target_pos
            if hasattr(base_env, "target_pos")
            else TARGET_POS,
            obstacle_positions,
            obs_velocities=obstacle_velocities
        )
    )

    expert_physical_action = np.asarray(
        [expert_v, expert_omega],
        dtype=np.float32
    )

    expert_normalized_action = np.asarray(
        [
            2.0 * expert_v / MAX_LINEAR_VELOCITY - 1.0,
            expert_omega / MAX_ANGULAR_VELOCITY
        ],
        dtype=np.float32
    )

    expert_normalized_action = np.clip(
        expert_normalized_action,
        -1.0,
        1.0
    )

    print()
    print("      ACTUAL EXPERT ACTION AT THIS STATE")
    print(
        f"      Physical   : "
        f"v={expert_physical_action[0]:.3f}, "
        f"omega={expert_physical_action[1]:.3f}"
    )
    print(
        f"      Normalized : "
        f"v={expert_normalized_action[0]:.3f}, "
        f"omega={expert_normalized_action[1]:.3f}"
    )

    # ---------------------------------------------------------------
    # Candidate actions: actual expert action + nearby perturbations
    # ---------------------------------------------------------------
    policy_action, _ = model.predict(
        state,
        deterministic=True
    )
    policy_action = np.clip(
        np.asarray(policy_action, dtype=np.float32),
        -1.0,
        1.0
    )

    candidate_actions = [
        ("EXPERT", expert_normalized_action),
        ("POLICY", policy_action)
    ]

    deltas = [
        np.array([0.05, 0.0], dtype=np.float32),
        np.array([-0.05, 0.0], dtype=np.float32),
        np.array([0.0, 0.05], dtype=np.float32),
        np.array([0.0, -0.05], dtype=np.float32),
        np.array([0.10, 0.0], dtype=np.float32),
        np.array([-0.10, 0.0], dtype=np.float32),
        np.array([0.0, 0.10], dtype=np.float32),
        np.array([0.0, -0.10], dtype=np.float32),
        np.array([0.10, 0.10], dtype=np.float32),
        np.array([0.10, -0.10], dtype=np.float32),
        np.array([-0.10, 0.10], dtype=np.float32),
        np.array([-0.10, -0.10], dtype=np.float32)
    ]

    for idx, delta in enumerate(deltas):
        candidate_actions.append(
            (
                f"EXPERT+{idx + 1}",
                np.clip(
                    expert_normalized_action + delta,
                    -1.0,
                    1.0
                )
            )
        )

    print()
    print(
        "      Action              AIRL f(s,a,s')"
        "      Env       Hybrid"
    )
    print("      " + "-" * 68)

    scores = []

    # ---------------------------------------------------------------
    # Save exact state
    # ---------------------------------------------------------------

    snapshot = _snapshot_env_state(
        base_env,
        env
    )

    for name, action in candidate_actions:

        _restore_env_state(
            base_env,
            env,
            snapshot
        )

        action = np.asarray(
            action,
            dtype=np.float32
        )

        # -----------------------------------------------------------
        # Execute candidate action
        # -----------------------------------------------------------

        next_obs, env_reward, terminated, truncated, info = (
            base_env.step(action)
        )

        done = bool(
            terminated or truncated
        )

        next_state = np.asarray(
            next_obs,
            dtype=np.float32
        )

        # -----------------------------------------------------------
        # IMPORTANT:
        # AIRL reward is transition-based.
        # Use s, a, s', done.
        # -----------------------------------------------------------

        airl_reward = float(
            discriminator.predict_reward(
                state,
                action,
                next_state,
                done
            )
        )

        airl_component = float(
            np.clip(
                airl_reward,
                -AIRL_REWARD_CLIP,
                AIRL_REWARD_CLIP
            )
        )

        env_component = float(
            env_reward
        )

        hybrid_reward = (
            ENV_WEIGHT * env_component
            + AIRL_REWARD_WEIGHT * airl_component
        )

        scores.append(
            {
                "name": name,
                "airl": airl_component,
                "env": env_component,
                "hybrid": hybrid_reward,
                "done": done,
                "next_state": next_state.copy()
            }
        )

        print(
            f"      {name:<20}"
            f"{airl_component:>13.3f}"
            f"{env_component:>10.3f}"
            f"{hybrid_reward:>11.3f}"
        )

    # ---------------------------------------------------------------
    # Restore original state
    # ---------------------------------------------------------------

    _restore_env_state(
        base_env,
        env,
        snapshot
    )

    try:
        p.removeState(
            snapshot["pybullet_state"],
            physicsClientId=base_env.physics_client
        )
    except Exception:
        pass

    # ---------------------------------------------------------------
    # Ranking
    # ---------------------------------------------------------------

    best_airl = max(
        scores,
        key=lambda x: x["airl"]
    )

    best_hybrid = best_airl

    expert_result = next(
        x for x in scores
        if x["name"] == "EXPERT"
    )

    policy_result = next(
        x for x in scores
        if x["name"] == "POLICY"
    )

    airl_ranked = sorted(
        scores,
        key=lambda x: x["airl"],
        reverse=True
    )

    expert_airl_rank = next(
        i + 1
        for i, x in enumerate(airl_ranked)
        if x["name"] == "EXPERT"
    )

    print()
    print("-" * 75)
    print("      DIAGNOSTIC RESULT")
    print("-" * 75)

    print(
        f"      Best AIRL action   : "
        f"{best_airl['name']} "
        f"({best_airl['airl']:.3f})"
    )

    print(
        f"      Best HYBRID action : "
        f"{best_hybrid['name']} "
        f"({best_hybrid['hybrid']:.3f})"
    )

    print(
        f"      Expert AIRL rank   : "
        f"{expert_airl_rank}/{len(scores)}"
    )

    print(
        f"      Expert AIRL reward : "
        f"{expert_result['airl']:.3f}"
    )

    print(
        f"      Policy AIRL        : "
        f"{next(x for x in scores if x['name'] == 'POLICY')['airl']:.3f}"
    )

    print()

    # ---------------------------------------------------------------
    # Tests
    # ---------------------------------------------------------------

    if expert_result["airl"] > policy_result["airl"]:
        print(
            "      AIRL TEST: PASS"
        )
        print(
            "      AIRL gives the expert action "
            "more reward than the current PPO action."
        )
    else:
        print(
            "      AIRL TEST: FAIL"
        )
        print(
            "      AIRL does NOT prefer the expert action "
            "over straight motion."
        )

    # ---------------------------------------------------------------
    # Stronger expert test
    # ---------------------------------------------------------------

    expert_is_best = (
        best_airl["name"] == "EXPERT"
    )

    if expert_is_best:
        print(
            "      EXPERT RANKING: PASS"
        )
        print(
            "      AIRL ranks the actual expert action highest."
        )
    else:
        print(
            "      EXPERT RANKING: FAIL"
        )
        print(
            f"      AIRL prefers '{best_airl['name']}' "
            f"over the expert action."
        )

    print()
    print(
        "      This diagnostic does NOT update AIRL or PPO."
    )
    print("-" * 75)

def load_expert_dataset():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Missing {DATASET_PATH}"
        )

    data = np.load(DATASET_PATH)

    required = (
        "observations",
        "actions_normalized",
        "next_observations",
        "dones"
    )

    missing = [
        key for key in required
        if key not in data
    ]

    if missing:
        raise ValueError(
            f"demo_dataset.npz is missing required fields: {missing}"
        )

    expert_states = data[
        "observations"
    ].astype(np.float32)

    expert_actions = data[
        "actions_normalized"
    ].astype(np.float32)

    expert_next_states = data[
        "next_observations"
    ].astype(np.float32)

    expert_dones = data[
        "dones"
    ].astype(np.float32).reshape(-1, 1)

    if expert_states.ndim != 2 or expert_states.shape[1] != STATE_DIM:
        raise ValueError(
            f"Expert observations must have shape (N, 28), "
            f"got {expert_states.shape}."
        )

    if expert_next_states.shape != expert_states.shape:
        raise ValueError(
            "observations and next_observations must have "
            "identical shapes."
        )

    if expert_actions.ndim != 2 or expert_actions.shape[1] != 2:
        raise ValueError(
            f"actions_normalized must have shape (N, 2), "
            f"got {expert_actions.shape}."
        )

    if (
        len(expert_states) != len(expert_actions)
        or len(expert_states) != len(expert_dones)
    ):
        raise ValueError(
            "observations, actions_normalized, "
            "next_observations and dones must have the same length."
        )

    if not os.path.exists(BC_WEIGHT_PATH):
        raise FileNotFoundError(
            "imitation_policy_weights.npz is required because "
            "AIRL and PPO use normalized observations."
        )

    bc_data = np.load(BC_WEIGHT_PATH)

    if "mean" not in bc_data or "std" not in bc_data:
        raise ValueError(
            "imitation_policy_weights.npz must contain both "
            "'mean' and 'std'."
        )

    mean = bc_data[
        "mean"
    ].astype(np.float32)

    std = (
        bc_data["std"].astype(np.float32)
        + 1e-6
    )

    if mean.shape != (STATE_DIM,) or std.shape != (STATE_DIM,):
        raise ValueError(
            f"Normalization mean/std must both have shape ({STATE_DIM},), "
            f"got {mean.shape} and {std.shape}."
        )

    # collect_dataset.py stores RAW observations.
    # AirlNavEnv normalizes agent observations with these same
    # statistics, so normalize expert s and s' exactly once here.
    expert_states = (
        expert_states - mean
    ) / std

    expert_next_states = (
        expert_next_states - mean
    ) / std

    print(
        "Expert observations normalized with BC mean/std."
    )

    print(
        "Expert actions loaded from actions_normalized."
    )

    return (
        expert_states,
        expert_actions,
        expert_next_states,
        expert_dones
    )


def behavior_clone_warmup(
    model,
    expert_states,
    expert_actions,
    epochs=20,
    batch_size=256,
    max_samples=16384
):
    n = min(len(expert_states), max_samples)

    if n == 0:
        return

    indices = np.arange(n)
    optimizer = torch.optim.Adam(
        model.policy.parameters(),
        lr=3e-4
    )

    model.policy.train()

    for epoch in range(epochs):
        np.random.shuffle(indices)
        losses = []

        for start in range(0, n, batch_size):
            idx = indices[start:start + batch_size]

            states_tensor, _ = (
                model.policy.obs_to_tensor(
                    expert_states[idx]
                )
            )

            actions_tensor = torch.as_tensor(
                expert_actions[idx],
                dtype=torch.float32,
                device=model.device
            )

            _, log_prob, _ = (
                model.policy.evaluate_actions(
                    states_tensor,
                    actions_tensor
                )
            )

            distribution = model.policy.get_distribution(
                states_tensor
            )

            mean_actions = distribution.distribution.mean

            bc_mse = torch.mean(
                (mean_actions - actions_tensor) ** 2
            )

            loss = -log_prob.mean() + 2.0 * bc_mse

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.policy.parameters(),
                0.5
            )

            optimizer.step()
            losses.append(float(loss.item()))

        print(
            f"      BC warmup epoch {epoch + 1}/{epochs}: "
            f"NLL={np.mean(losses):.4f}"
        )

    model.policy.eval()




def train_discriminator(
    discriminator,
    model,
    expert_states,
    expert_actions,
    expert_next_states,
    expert_dones,
    agent_states,
    agent_actions,
    agent_next_states,
    agent_dones,
    agent_log_probs
):

    losses = []
    expert_losses = []
    agent_losses = []

    n_expert = len(
        expert_states
    )

    n_agent = len(
        agent_states
    )

    if (
        n_expert == 0
        or n_agent == 0
    ):

        raise ValueError(
            "Expert or agent dataset is empty."
        )

    expert_log_probs = (
        get_policy_log_probs(
            model,
            expert_states,
            expert_actions
        )
    )

    for _ in range(
        DISCRIMINATOR_UPDATES
    ):

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
            expert_states[
                expert_idx
            ],
            expert_actions[
                expert_idx
            ],
            expert_next_states[
                expert_idx
            ],
            expert_dones[
                expert_idx
            ],
            expert_log_probs[
                expert_idx
            ],
            agent_states[
                agent_idx
            ],
            agent_actions[
                agent_idx
            ],
            agent_next_states[
                agent_idx
            ],
            agent_dones[
                agent_idx
            ],
            agent_log_probs[
                agent_idx
            ]
        )

        losses.append(
            float(loss[0])
        )

        expert_losses.append(
            float(loss[1])
        )

        agent_losses.append(
            float(loss[2])
        )

    return (
        float(np.mean(losses)),
        float(np.mean(expert_losses)),
        float(np.mean(agent_losses))
    )


def evaluate_policy(
    model,
    env,
    episodes=7
):

    env.set_discriminator(
        None,
        False
    )

    base_env = env.unwrapped

    results = []

    for _ in range(episodes):

        obs, _ = env.reset()

        previous_pos = (
            base_env.robot_pos.copy()
        )

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

            next_obs, _, terminated, truncated, info = \
                env.step(action)

            current_pos = (
                base_env.robot_pos.copy()
            )

            path_length += np.linalg.norm(
                current_pos - previous_pos
            )

            previous_pos = (
                current_pos
            )

            steps += 1

            final_distance = info[
                "dist"
            ]

            goal |= bool(
                info["goal_reached"]
            )

            collision |= bool(
                info["collision"]
            )

            wall |= bool(
                info["wall_hit"]
            )

            obs = next_obs

            if (
                terminated
                or truncated
            ):
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

        "goal_success_rate":
            100.0 * np.mean(
                [x["goal"] for x in results]
            ),

        "collision_rate":
            100.0 * np.mean(
                [x["collision"] for x in results]
            ),

        "wall_hit_rate":
            100.0 * np.mean(
                [x["wall"] for x in results]
            ),

        "average_final_distance":
            float(
                np.mean(
                    [
                        x["final_distance"]
                        for x in results
                    ]
                )
            ),

        "average_episode_length":
            float(
                np.mean(
                    [
                        x["steps"]
                        for x in results
                    ]
                )
            ),

        "average_path_length":
            float(
                np.mean(
                    [
                        x["path_length"]
                        for x in results
                    ]
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

        writer.writerows(
            history
        )


def train_airl_ppo():

    print("=" * 75)

    print(
        "              AIRL + PPO TRAINING"
    )

    print("=" * 75)

    (
        expert_states,
        expert_actions,
        expert_next_states,
        expert_dones
    ) = load_expert_dataset()



    print(
        f"Expert samples : "
        f"{len(expert_states)}"
    )

    base_env = AirlNavEnv(
        render_gui=False,
        normalize_obs=True
    )

    discriminator = AIRLDiscriminator(
        state_dim=28,
        action_dim=2,
        hidden_dim=128,
        gamma=0.99,
        lr=3e-4
    )

    discriminator.loss_fn = LabelSmoothingBCEWithLogitsLoss(
        DISCRIMINATOR_LABEL_SMOOTHING
    )

    env = AIRLRewardWrapper(
        base_env,
        discriminator=None,
        use_airl_reward=False,
        reward_scale=AIRL_REWARD_SCALE,
        reward_clip=AIRL_REWARD_CLIP,
        airl_weight=AIRL_REWARD_WEIGHT,
        env_weight = ENV_WEIGHT
    )

    policy_kwargs = dict(
        net_arch=dict(
            pi=[128, 64],
            vf=[128, 64]
        ),
        activation_fn=torch.nn.ReLU,
        log_std_init=-1.5
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
        ent_coef=0.001,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1
    )

    print()
    print("      AIRL BC warm-start")
    print("      Initializing PPO policy from expert actions...")

    behavior_clone_warmup(
        model,
        expert_states,
        expert_actions,
        epochs=20,
        batch_size=256,
        max_samples=16384
    )

    history = []

    print(f"AIRL reward scale : {AIRL_REWARD_SCALE}")
    print(f"Reward clip       : +/-{AIRL_REWARD_CLIP}")
    print(f"AIRL reward weight: {AIRL_REWARD_WEIGHT}")
    print(f"Environment weight: {ENV_WEIGHT}")
    print(f"Discriminator updates/iter: {DISCRIMINATOR_UPDATES}")
    print("AIRL reward: f(s,a,s') = g(s,a) + gamma*h(s') - h(s)")
    print("PPO reward: AIRL_WEIGHT * clipped f only")

    steps_per_iteration = (
        TOTAL_TIMESTEPS
        // AIRL_ITERATIONS
    )

    for iteration in range(
        AIRL_ITERATIONS
    ):

        print()
        print("=" * 75)

        print(
            f"AIRL ITERATION "
            f"{iteration + 1}/"
            f"{AIRL_ITERATIONS}"
        )

        print("=" * 75)

        print(
            "[1/4] Collecting PPO "
            "agent transitions..."
        )

        env.set_discriminator(
            None,
            False
        )

        (
            agent_states,
            agent_actions,
            agent_next_states,
            agent_dones,
            agent_log_probs,
            agent_env_rewards
        ) = collect_agent_data(
            model,
            env,
            DISCRIMINATOR_ROLLOUT_SIZE
        )

        print(
            f"      Agent samples: "
            f"{len(agent_states)}"
        )

        print(
            "[2/4] Training AIRL "
            "discriminator..."
        )

        (
            d_loss,
            expert_loss,
            agent_loss
        ) = train_discriminator(
            discriminator,
            model,
            expert_states,
            expert_actions,
            expert_next_states,
            expert_dones,
            agent_states,
            agent_actions,
            agent_next_states,
            agent_dones,
            agent_log_probs
        )

        expert_log_probs = (
            get_policy_log_probs(
                model,
                expert_states[:1000],
                expert_actions[:1000]
            )
        )

        expert_log_prob_mean = float(
            np.mean(expert_log_probs)
        )

        agent_log_prob_mean = float(
            np.mean(agent_log_probs[:1000])
        )

        d_expert = (
            discriminator.discriminator_probability(
                expert_states[:1000],
                expert_actions[:1000],
                expert_next_states[:1000],
                expert_dones[:1000],
                expert_log_probs
            )
        )

        d_agent = (
            discriminator.discriminator_probability(
                agent_states[:1000],
                agent_actions[:1000],
                agent_next_states[:1000],
                agent_dones[:1000],
                agent_log_probs[:1000]
            )
        )

        raw_reward = np.asarray(
            discriminator.predict_reward(
                agent_states[:1000],
                agent_actions[:1000],
                agent_next_states[:1000],
                agent_dones[:1000]
            ),
            dtype=np.float32
        ).reshape(-1)

        scaled_reward = raw_reward * AIRL_REWARD_SCALE
        env_sample = agent_env_rewards[:1000]
        env_component = env_sample.copy()
        clipped_airl = np.clip(scaled_reward, -AIRL_REWARD_CLIP, AIRL_REWARD_CLIP)
        hybrid_reward = AIRL_REWARD_WEIGHT * clipped_airl

        print(
            f"      D loss       : "
            f"{d_loss:.4f}"
        )

        print(
            f"      Expert BCE   : "
            f"{expert_loss:.4f}"
        )

        print(
            f"      Agent BCE    : "
            f"{agent_loss:.4f}"
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
            f"      Expert log pi : "
            f"{expert_log_prob_mean:.4f}"
        )

        print(
            f"      Agent log pi  : "
            f"{agent_log_prob_mean:.4f}"
        )

        print(
            f"      AIRL f(s,a,s') : "
            f"mean={np.mean(raw_reward):.4f}, "
            f"min={np.min(raw_reward):.4f}, "
            f"max={np.max(raw_reward):.4f}"
        )

        print(
            f"      AIRL scaled  : "
            f"mean={np.mean(scaled_reward):.4f}, "
            f"min={np.min(scaled_reward):.4f}, "
            f"max={np.max(scaled_reward):.4f}"
        )

        print(
            f"      Env reward   : mean={np.mean(env_sample):.4f}, "
            f"min={np.min(env_sample):.4f}, "
            f"max={np.max(env_sample):.4f}"
        )

        print(
            f"      Env component: mean={np.mean(env_component):.4f}, "
            f"min={np.min(env_component):.4f}, "
            f"max={np.max(env_component):.4f}"
        )

        print(
            f"      Hybrid PPO   : "
            f"mean={np.mean(hybrid_reward):.4f}, "
            f"min={np.min(hybrid_reward):.4f}, "
            f"max={np.max(hybrid_reward):.4f}"
        )

        print(
            f"      Hybrid at -20: "
            f"{100.0 * np.mean(hybrid_reward <= -19.999):.2f}%"
        )

        print(
            f"      Hybrid at +20: "
            f"{100.0 * np.mean(hybrid_reward >= 19.999):.2f}%"
        )

        run_airl_action_diagnostic(
            model,
            env,
            discriminator
        )

        print(
            "[3/4] PPO training with "
            "AIRL reward..."
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
            "iteration":
                iteration + 1,
            **metrics
        }

        history.append(
            row
        )

        save_metrics(
            history
        )

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

        # Save this iteration's PPO model separately
        iteration_model_path = (
            f"{MODEL_ITER_PREFIX}{iteration + 1}"
        )

        model.save(
            iteration_model_path
        )

        # Also save the latest model
        model.save(
            MODEL_PATH
        )

        # Save discriminator for this iteration
        iteration_discriminator_path = (
            f"airl_ppo_discriminator_iter_{iteration + 1}.pth"
        )

        torch.save(
            discriminator.state_dict(),
            iteration_discriminator_path
        )

        # Keep the latest discriminator as well
        torch.save(
            discriminator.state_dict(),
            DISCRIMINATOR_PATH
        )

        print(
            f"      Saved PPO model: "
            f"{iteration_model_path}.zip"
        )

        print(
            f"      Saved discriminator: "
            f"{iteration_discriminator_path}"
        )

    env.close()

    print()
    print("=" * 75)
    print(
        "              AIRL + PPO COMPLETED"
    )
    print("=" * 75)

    print(
        f"Model          : "
        f"{MODEL_PATH}.zip"
    )

    print(
        f"Discriminator  : "
        f"{DISCRIMINATOR_PATH}"
    )

    print(
        f"Metrics        : "
        f"{METRICS_PATH}"
    )


if __name__ == "__main__":
    train_airl_ppo()