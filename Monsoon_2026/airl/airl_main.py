import os
import sys
import time
import csv

import numpy as np
import matplotlib.pyplot as plt
import pybullet as p
import pybullet_data

from stable_baselines3 import PPO

from feature_extractor import CNNFeatureExtractor
from train_imitation import ImitationPolicyNet
from obstacle_manager import ObstacleManager


# ============================================================
# PARAMETERS
# ============================================================

IMG_WIDTH = 320
IMG_HEIGHT = 240
FOV = 90

DT = 1.0 / 60.0

ARENA_HALF_LENGTH = 16.0
ARENA_HALF_WIDTH = 8.0

START_POS = np.array(
    [-12.0, 0.0],
    dtype=np.float32
)

TARGET_POS = np.array(
    [12.0, 0.0],
    dtype=np.float32
)

ROBOT_Z = 0.1

GOAL_THRESHOLD = 0.50
COLLISION_THRESHOLD = 0.64
WALL_THRESHOLD = 0.60

MAX_LINEAR_VELOCITY = 4.0
MAX_ANGULAR_VELOCITY = 2.5

TOTAL_STEPS = 1500


# ============================================================
# BEST MODEL SELECTION
# ============================================================

METRICS_FILE = "airl_ppo_metrics.csv"

MODEL_PREFIX = "airl_ppo_model_iter_"


def select_best_model():
    """
    Select the best PPO iteration.

    Priority:
        1. 100% goal success.
        2. Otherwise, safest policy with minimum final distance.
        3. Otherwise, minimum final distance overall.

    Returns:
        model_path, selected_metrics
    """

    if not os.path.exists(METRICS_FILE):
        raise FileNotFoundError(
            f"\nMissing {METRICS_FILE}\n"
            f"The training script must save iteration metrics "
            f"before airl_main.py can automatically select "
            f"the best policy."
        )

    rows = []

    with open(METRICS_FILE, "r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:

            try:
                iteration = int(float(row["iteration"]))

                goal_success = float(
                    row["goal_success_rate"]
                )

                collision = float(
                    row["collision_rate"]
                )

                wall_hit = float(
                    row["wall_hit_rate"]
                )

                final_distance = float(
                    row["average_final_distance"]
                )

                path_length = float(
                    row.get(
                        "average_path_length",
                        float("inf")
                    )
                )

                rows.append({
                    "iteration": iteration,
                    "goal_success_rate": goal_success,
                    "collision_rate": collision,
                    "wall_hit_rate": wall_hit,
                    "average_final_distance": final_distance,
                    "average_path_length": path_length
                })

            except (ValueError, KeyError):
                continue

    if not rows:
        raise RuntimeError(
            f"No valid training metrics found in "
            f"{METRICS_FILE}."
        )

    # --------------------------------------------------------
    # First priority:
    # Any policy with 100% goal success
    # --------------------------------------------------------

    successful = [
        r for r in rows
        if r["goal_success_rate"] >= 100.0
    ]

    if successful:

        best = min(
            successful,
            key=lambda r: (
                r["average_final_distance"],
                r["collision_rate"],
                r["wall_hit_rate"],
                r["average_path_length"]
            )
        )

        selection_reason = (
            "100% goal-success policy with the "
            "smallest final distance"
        )

    else:

        # ----------------------------------------------------
        # Second priority:
        # Safe policies
        # ----------------------------------------------------

        safe = [
            r for r in rows
            if r["collision_rate"] == 0.0
            and r["wall_hit_rate"] == 0.0
        ]

        if safe:

            best = min(
                safe,
                key=lambda r: (
                    r["average_final_distance"],
                    -r["goal_success_rate"],
                    r["average_path_length"]
                )
            )

            selection_reason = (
                "collision-free and wall-free policy "
                "with the smallest final distance"
            )

        else:

            # ------------------------------------------------
            # Third priority:
            # Lowest final distance overall
            # ------------------------------------------------

            best = min(
                rows,
                key=lambda r: (
                    r["average_final_distance"],
                    -r["goal_success_rate"],
                    r["collision_rate"],
                    r["wall_hit_rate"]
                )
            )

            selection_reason = (
                "policy with the smallest final distance "
                "because no completely safe policy exists"
            )

    iteration = best["iteration"]

    candidate_paths = [
        f"{MODEL_PREFIX}{iteration}.zip",
        f"{MODEL_PREFIX}{iteration}"
    ]

    model_path = None

    for path in candidate_paths:

        if os.path.exists(path):
            model_path = path
            break

    if model_path is None:

        raise FileNotFoundError(
            f"\nMetrics selected iteration {iteration}, "
            f"but its PPO model was not found.\n\n"
            f"Expected:\n"
            f"  {MODEL_PREFIX}{iteration}.zip\n"
        )

    # --------------------------------------------------------
    # Print selection
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("  BEST PPO POLICY SELECTION")
    print("=" * 70)

    print(
        f"\n  Selected iteration : {iteration}"
    )

    print(
        f"  Goal success       : "
        f"{best['goal_success_rate']:.2f}%"
    )

    print(
        f"  Collision          : "
        f"{best['collision_rate']:.2f}%"
    )

    print(
        f"  Wall hit           : "
        f"{best['wall_hit_rate']:.2f}%"
    )

    print(
        f"  Final distance     : "
        f"{best['average_final_distance']:.3f} m"
    )

    print(
        f"  Path length        : "
        f"{best['average_path_length']:.3f} m"
    )

    print(
        f"\n  Selection reason   : "
        f"{selection_reason}"
    )

    print(
        f"\n  Loading model      : "
        f"{model_path}"
    )

    print("=" * 70)

    return model_path, best


# ============================================================
# REWARD
# ============================================================

def compute_step_reward(
    dist,
    prev_dist,
    collision,
    wall_hit,
    v,
    omega,
    prev_v,
    prev_omega,
    goal_reached
):

    r_progress = (prev_dist - dist) * 10.0

    r_goal = 100.0 if goal_reached else 0.0

    r_collision = -50.0 if collision else 0.0

    r_wall = -50.0 if wall_hit else 0.0

    r_speed = -0.02 * (v ** 2)

    r_smooth = -0.05 * (
        (v - prev_v) ** 2
        + (omega - prev_omega) ** 2
    )

    total_reward = (
        r_progress
        + r_goal
        + r_collision
        + r_wall
        + r_speed
        + r_smooth
    )

    return total_reward, {
        "progress": r_progress,
        "goal": r_goal,
        "collision": r_collision,
        "wall": r_wall,
        "speed": r_speed,
        "smooth": r_smooth
    }


# ============================================================
# ARENA
# ============================================================

def create_arena_walls(
    half_length,
    half_width,
    wall_height=1.0,
    wall_thickness=0.2
):

    wall_ids = []

    z_center = wall_height / 2.0

    wall_specs = [
        (
            [half_length, wall_thickness / 2.0, wall_height / 2.0],
            [0, half_width, z_center]
        ),
        (
            [half_length, wall_thickness / 2.0, wall_height / 2.0],
            [0, -half_width, z_center]
        ),
        (
            [wall_thickness / 2.0, half_width, wall_height / 2.0],
            [half_length, 0, z_center]
        ),
        (
            [wall_thickness / 2.0, half_width, wall_height / 2.0],
            [-half_length, 0, z_center]
        )
    ]

    for half_extents, pos in wall_specs:

        visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=[0.6, 0.6, 0.6, 1]
        )

        collision_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=half_extents
        )

        wall_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_shape,
            baseCollisionShapeIndex=collision_shape,
            basePosition=pos
        )

        wall_ids.append(wall_id)

    return wall_ids


# ============================================================
# ROBOT
# ============================================================

def create_robot():

    visual_shape = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[0.4, 0.4, 0.2],
        rgbaColor=[0, 0, 1, 1]
    )

    collision_shape = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[0.4, 0.4, 0.2]
    )

    robot_id = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=visual_shape,
        baseCollisionShapeIndex=collision_shape,
        basePosition=[
            START_POS[0],
            START_POS[1],
            ROBOT_Z
        ]
    )

    return robot_id


# ============================================================
# GOAL
# ============================================================

def create_goal():

    visual_shape = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=0.4,
        rgbaColor=[0, 1, 0, 1]
    )

    goal_id = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=visual_shape,
        basePosition=[
            TARGET_POS[0],
            TARGET_POS[1],
            0.0
        ]
    )

    return goal_id


# ============================================================
# OBSTACLES
# ============================================================

def get_15_obstacle_initial_positions():

    pursuers = [
        [-7.0, 3.0, 0.4],
        [-7.0, -3.0, 0.4],
        [-3.6, 1.6, 0.4],
        [-3.6, -1.6, 0.4],
        [-1.0, 3.6, 0.4]
    ]

    defenders = [
        [9.6, 2.4, 0.4],
        [9.6, -2.4, 0.4],
        [11.0, 1.0, 0.4],
        [11.0, -1.0, 0.4],
        [12.4, 3.0, 0.4]
    ]

    roamers = [
        [1.6, 4.4, 0.4],
        [1.6, -4.4, 0.4],
        [5.0, 3.6, 0.4],
        [5.0, -3.6, 0.4],
        [7.6, 0.0, 0.4]
    ]

    return pursuers + defenders + roamers


def create_obstacles(initial_positions):

    obstacle_ids = []

    for pos in initial_positions:

        visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.24, 0.24, 0.4],
            rgbaColor=[1, 0, 0, 1]
        )

        collision_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[0.24, 0.24, 0.4]
        )

        obstacle_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_shape,
            baseCollisionShapeIndex=collision_shape,
            basePosition=pos
        )

        obstacle_ids.append(obstacle_id)

    return obstacle_ids


# ============================================================
# PYBULLET SYNC
# ============================================================

def sync_robot_to_pybullet(
    robot_id,
    robot_pos,
    robot_yaw
):

    quat = p.getQuaternionFromEuler(
        [0.0, 0.0, robot_yaw]
    )

    p.resetBasePositionAndOrientation(
        robot_id,
        [
            float(robot_pos[0]),
            float(robot_pos[1]),
            ROBOT_Z
        ],
        quat
    )


def sync_obstacles_to_pybullet(
    obstacle_ids,
    obstacle_positions
):

    for idx, obstacle_id in enumerate(obstacle_ids):

        p.resetBasePositionAndOrientation(
            obstacle_id,
            [
                float(obstacle_positions[idx][0]),
                float(obstacle_positions[idx][1]),
                float(obstacle_positions[idx][2])
            ],
            [0, 0, 0, 1]
        )


# ============================================================
# COLLISION
# ============================================================

def check_collision(
    robot_pos,
    obstacle_positions
):

    robot_xy = np.asarray(
        robot_pos[:2],
        dtype=np.float32
    )

    for obs in obstacle_positions:

        obs_xy = np.asarray(
            obs[:2],
            dtype=np.float32
        )

        if np.linalg.norm(
            robot_xy - obs_xy
        ) < COLLISION_THRESHOLD:

            return True

    return False


def check_wall_hit(robot_pos):

    x = robot_pos[0]
    y = robot_pos[1]

    return (
        abs(x) > ARENA_HALF_LENGTH - WALL_THRESHOLD
        or
        abs(y) > ARENA_HALF_WIDTH - WALL_THRESHOLD
    )


# ============================================================
# VISION
# ============================================================

class PyBulletVisionSystem:

    def __init__(self, robot_id):

        self.robot_id = robot_id

        self.width = IMG_WIDTH
        self.height = IMG_HEIGHT
        self.fov = FOV

        self.extractor = CNNFeatureExtractor(
            img_width=IMG_WIDTH,
            img_height=IMG_HEIGHT,
            fov_deg=FOV,
            stack_frames=2
        )

    def reset(self):

        self.extractor.reset()

    def capture_observation(self):

        pos, orn = p.getBasePositionAndOrientation(
            self.robot_id
        )

        rot_matrix = np.array(
            p.getMatrixFromQuaternion(orn)
        ).reshape(3, 3)

        cam_pos = (
            np.array(pos)
            + np.array([0.0, 0.0, 0.35])
        )

        target_cam = (
            cam_pos
            + rot_matrix.dot(
                np.array([1.0, 0.0, 0.0])
            )
        )

        view_matrix = p.computeViewMatrix(
            cam_pos,
            target_cam,
            [0, 0, 1]
        )

        proj_matrix = p.computeProjectionMatrixFOV(
            self.fov,
            self.width / self.height,
            0.02,
            20.0
        )

        _, _, rgb, _, _ = p.getCameraImage(
            self.width,
            self.height,
            view_matrix,
            proj_matrix
        )

        rgba = np.array(
            rgb,
            dtype=np.uint8
        ).reshape(
            (self.height, self.width, 4)
        )

        observation = (
            self.extractor.extract_observation(
                rgba
            )
        )

        return rgba, observation


# ============================================================
# NORMALIZATION
# ============================================================

def load_normalization():

    weights_path = "imitation_policy_weights.npz"

    if not os.path.exists(weights_path):

        raise FileNotFoundError(
            f"Missing {weights_path}"
        )

    data = np.load(weights_path)

    if "mean" not in data or "std" not in data:

        raise ValueError(
            "imitation_policy_weights.npz "
            "does not contain mean/std."
        )

    mean = data["mean"].astype(
        np.float32
    )

    std = data["std"].astype(
        np.float32
    )

    std = np.maximum(
        std,
        1e-6
    )

    if mean.shape != (32,):

        raise ValueError(
            f"Expected mean shape (32,), "
            f"got {mean.shape}"
        )

    if std.shape != (32,):

        raise ValueError(
            f"Expected std shape (32,), "
            f"got {std.shape}"
        )

    return mean, std


def normalize_observation(
    raw_observation,
    mean,
    std
):

    normalized = (
        raw_observation - mean
    ) / std

    return normalized.astype(
        np.float32
    )


# ============================================================
# ACTION CONVERSION
# ============================================================

def normalized_to_physical_action(
    action
):

    action = np.asarray(
        action,
        dtype=np.float32
    )

    action = np.clip(
        action,
        -1.0,
        1.0
    )

    v = (
        0.5
        * (action[0] + 1.0)
        * MAX_LINEAR_VELOCITY
    )

    omega = (
        action[1]
        * MAX_ANGULAR_VELOCITY
    )

    return float(v), float(omega)


# ============================================================
# SIMULATION
# ============================================================

def run_simulation(gui=True):

    # --------------------------------------------------------
    # Automatically select best trained PPO policy
    # --------------------------------------------------------

    model_path, selected_metrics = (
        select_best_model()
    )

    obs_mean, obs_std = load_normalization()

    print(
        "\n[INFO] Loaded observation normalization."
    )

    print(
        f"[INFO] Mean shape: {obs_mean.shape}"
    )

    print(
        f"[INFO] Std shape : {obs_std.shape}"
    )

    print(
        f"[INFO] Std range : "
        f"{obs_std.min():.6f} - "
        f"{obs_std.max():.6f}"
    )

    print(
        "\n[INFO] Loading BEST PPO model..."
    )

    sb3_ppo_model = PPO.load(
        model_path
    )

    print(
        "[INFO] Best PPO model loaded successfully."
    )

    # --------------------------------------------------------
    # PyBullet
    # --------------------------------------------------------

    if gui:
        p.connect(p.GUI)
    else:
        p.connect(p.DIRECT)

    p.setAdditionalSearchPath(
        pybullet_data.getDataPath()
    )

    p.setGravity(
        0,
        0,
        -9.8
    )

    p.loadURDF(
        "plane.urdf"
    )

    create_arena_walls(
        ARENA_HALF_LENGTH,
        ARENA_HALF_WIDTH
    )

    robot_id = create_robot()

    create_goal()

    initial_obs = (
        get_15_obstacle_initial_positions()
    )

    obs_ids = create_obstacles(
        initial_obs
    )

    obs_mgr = ObstacleManager(
        initial_obs,
        speed=0.40,
        separation_dist=1.30,
        arena_bounds=(
            ARENA_HALF_LENGTH - 1.0,
            ARENA_HALF_WIDTH - 1.0
        ),
        target_pos=TARGET_POS
    )

    vision = PyBulletVisionSystem(
        robot_id
    )

    vision.reset()

    robot_pos = START_POS.copy()

    robot_yaw = 0.0

    trajectory = {
        "x": [],
        "y": []
    }

    actions_policy = {
        "v": [],
        "omega": []
    }

    actions_normalized = {
        "v": [],
        "omega": []
    }

    rewards_history = {
        "total": [],
        "progress": [],
        "goal": [],
        "collision": [],
        "wall": [],
        "speed": [],
        "smooth": []
    }

    prev_dist = float(
        np.linalg.norm(
            robot_pos - TARGET_POS
        )
    )

    prev_v = 0.0
    prev_omega = 0.0

    print()
    print("=" * 70)
    print(
        "  Running BEST PPO Autonomous Navigation"
    )
    print(
        f"  Selected iteration: "
        f"{selected_metrics['iteration']}"
    )
    print(
        f"  Training final distance: "
        f"{selected_metrics['average_final_distance']:.3f} m"
    )
    print("=" * 70)

    final_status = "TIMEOUT"

    dist = prev_dist

    current_obs = initial_obs

    for step in range(TOTAL_STEPS):

        current_obs = obs_mgr.update(
            robot_pos,
            step_count=step,
            dt=DT
        )

        sync_robot_to_pybullet(
            robot_id,
            robot_pos,
            robot_yaw
        )

        sync_obstacles_to_pybullet(
            obs_ids,
            current_obs
        )

        p.stepSimulation()

        _, raw_state = (
            vision.capture_observation()
        )

        if raw_state.shape != (32,):

            raise ValueError(
                f"Expected 32-D observation, "
                f"got {raw_state.shape}"
            )

        state = normalize_observation(
            raw_state,
            obs_mean,
            obs_std
        )

        action_pred, _ = (
            sb3_ppo_model.predict(
                state,
                deterministic=True
            )
        )

        action_pred = np.asarray(
            action_pred,
            dtype=np.float32
        ).reshape(-1)

        if action_pred.shape[0] != 2:

            raise ValueError(
                f"Expected 2-D action, "
                f"got {action_pred.shape}"
            )

        action_pred = np.clip(
            action_pred,
            -1.0,
            1.0
        )

        v_pol, omega_pol = (
            normalized_to_physical_action(
                action_pred
            )
        )

        MAX_OMEGA_CHANGE = 0.75
        OMEGA_SMOOTHING = 0.75

        omega_delta = np.clip(
            omega_pol - prev_omega,
            -MAX_OMEGA_CHANGE,
            MAX_OMEGA_CHANGE
        )

        omega_used = (
            (1.0 - OMEGA_SMOOTHING)
            * prev_omega
            +
            OMEGA_SMOOTHING
            * (
                prev_omega
                + omega_delta
            )
        )

        v_exec = v_pol
        omega_exec = omega_used

        robot_pos = (
            robot_pos
            +
            v_exec
            *
            np.array(
                [
                    np.cos(robot_yaw),
                    np.sin(robot_yaw)
                ],
                dtype=np.float32
            )
            *
            DT
        )

        robot_yaw = (
            robot_yaw
            + omega_exec * DT
        )

        robot_yaw = np.arctan2(
            np.sin(robot_yaw),
            np.cos(robot_yaw)
        )

        sync_robot_to_pybullet(
            robot_id,
            robot_pos,
            robot_yaw
        )

        dist = np.linalg.norm(
            robot_pos - TARGET_POS
        )

        collision = check_collision(
            robot_pos,
            current_obs
        )

        wall_hit = check_wall_hit(
            robot_pos
        )

        goal_reached = (
            dist < GOAL_THRESHOLD
        )

        r_total, r_breakdown = (
            compute_step_reward(
                dist,
                prev_dist,
                collision,
                wall_hit,
                v_exec,
                omega_exec,
                prev_v,
                prev_omega,
                goal_reached
            )
        )

        trajectory["x"].append(
            robot_pos[0]
        )

        trajectory["y"].append(
            robot_pos[1]
        )

        actions_policy["v"].append(
            v_pol
        )

        actions_policy["omega"].append(
            omega_pol
        )

        actions_normalized["v"].append(
            action_pred[0]
        )

        actions_normalized["omega"].append(
            action_pred[1]
        )

        rewards_history["total"].append(
            r_total
        )

        for key in r_breakdown:

            rewards_history[key].append(
                r_breakdown[key]
            )

        prev_dist = dist
        prev_v = v_exec
        prev_omega = omega_exec

        if gui:

            p.resetDebugVisualizerCamera(
                cameraDistance=20,
                cameraYaw=0,
                cameraPitch=-60,
                cameraTargetPosition=[
                    robot_pos[0],
                    robot_pos[1],
                    0
                ]
            )

            time.sleep(
                1.0 / 240.0
            )

        if step % 100 == 0:

            print(
                f"Step {step:4d} | "
                f"Pos: ({robot_pos[0]:6.2f}, "
                f"{robot_pos[1]:6.2f}) | "
                f"Dist: {dist:6.2f} | "
                f"v_norm: {action_pred[0]:6.3f} | "
                f"ω_norm: {action_pred[1]:6.3f} | "
                f"v: {v_pol:5.2f} | "
                f"ω: {omega_pol:6.2f}"
            )

        if goal_reached:

            final_status = "SUCCESS"

            print(
                "\n[SUCCESS] Goal reached!"
            )

            print(
                f"Step       : {step}"
            )

            print(
                f"Final Pos  : "
                f"({robot_pos[0]:.3f}, "
                f"{robot_pos[1]:.3f})"
            )

            print(
                f"Final Dist : {dist:.3f} m"
            )

            break

        if collision:

            final_status = "COLLISION"

            print(
                "\n[COLLISION] "
                f"Robot collided at step {step}."
            )

            print(
                f"Position: "
                f"({robot_pos[0]:.3f}, "
                f"{robot_pos[1]:.3f})"
            )

            break

        if wall_hit:

            final_status = "WALL_HIT"

            print(
                "\n[WALL HIT] "
                f"Robot hit arena boundary at step {step}."
            )

            print(
                f"Position: "
                f"({robot_pos[0]:.3f}, "
                f"{robot_pos[1]:.3f})"
            )

            break

    p.disconnect()

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("  FINAL EVALUATION SUMMARY")
    print("=" * 70)

    print(
        f"Selected iteration : "
        f"{selected_metrics['iteration']}"
    )

    print(
        f"Status             : "
        f"{final_status}"
    )

    print(
        f"Steps              : "
        f"{len(trajectory['x'])}"
    )

    print(
        f"Final distance     : "
        f"{dist:.3f} m"
    )

    print(
        f"Final position     : "
        f"({robot_pos[0]:.3f}, "
        f"{robot_pos[1]:.3f})"
    )

    if len(trajectory["x"]) > 1:

        dx = np.diff(
            trajectory["x"]
        )

        dy = np.diff(
            trajectory["y"]
        )

        path_length = float(
            np.sum(
                np.sqrt(
                    dx ** 2
                    +
                    dy ** 2
                )
            )
        )

    else:

        path_length = 0.0

    print(
        f"Path length        : "
        f"{path_length:.3f} m"
    )

    print("=" * 70)

    # ========================================================
    # PLOTS
    # ========================================================

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5)
    )

    axes[0].plot(
        trajectory["x"],
        trajectory["y"],
        'b-',
        linewidth=2,
        label="Robot Trajectory"
    )

    for obs in current_obs:

        axes[0].plot(
            obs[0],
            obs[1],
            'rs',
            markersize=6
        )

    axes[0].plot(
        START_POS[0],
        START_POS[1],
        'k^',
        markersize=12,
        label="Start"
    )

    axes[0].plot(
        TARGET_POS[0],
        TARGET_POS[1],
        'g*',
        markersize=15,
        label="Goal"
    )

    axes[0].add_patch(
        plt.Rectangle(
            (
                -ARENA_HALF_LENGTH,
                -ARENA_HALF_WIDTH
            ),
            2 * ARENA_HALF_LENGTH,
            2 * ARENA_HALF_WIDTH,
            fill=False,
            edgecolor="gray",
            linestyle="--",
            linewidth=1.5
        )
    )

    axes[0].set_title(
        "Best PPO Navigation Trajectory"
    )

    axes[0].set_xlabel(
        "X (m)"
    )

    axes[0].set_ylabel(
        "Y (m)"
    )

    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(
        actions_policy["v"],
        label="Linear Speed (v)"
    )

    axes[1].plot(
        actions_policy["omega"],
        label="Angular Speed (ω)"
    )

    axes[1].set_title(
        "Physical Actions"
    )

    axes[1].set_xlabel(
        "Step"
    )

    axes[1].set_ylabel(
        "Action"
    )

    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(
        np.cumsum(
            rewards_history["total"]
        ),
        label="Cumulative Total",
        linewidth=2
    )

    axes[2].plot(
        np.cumsum(
            rewards_history["progress"]
        ),
        "--",
        label="Progress"
    )

    axes[2].plot(
        np.cumsum(
            rewards_history["wall"]
        ),
        "--",
        label="Wall Penalty"
    )

    axes[2].plot(
        np.cumsum(
            rewards_history["collision"]
        ),
        "--",
        label="Obstacle Penalty"
    )

    axes[2].plot(
        np.cumsum(
            rewards_history["speed"]
        ),
        "--",
        label="Speed Penalty"
    )

    axes[2].plot(
        np.cumsum(
            rewards_history["smooth"]
        ),
        "--",
        label="Smoothness"
    )

    axes[2].set_title(
        "Cumulative Evaluation Rewards"
    )

    axes[2].set_xlabel(
        "Step"
    )

    axes[2].set_ylabel(
        "Reward"
    )

    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()

    plt.savefig(
        "gail_evaluation_results.png",
        dpi=200
    )

    print(
        "\n[INFO] Saved evaluation plot to "
        "'gail_evaluation_results.png'"
    )

    if gui:
        plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    use_gui = (
        "--no-gui"
        not in sys.argv
    )

    run_simulation(
        gui=use_gui
    )