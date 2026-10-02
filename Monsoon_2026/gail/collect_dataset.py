

import numpy as np
import pybullet as p
import pybullet_data

from feature_extractor import CNNFeatureExtractor
from expert_controller import ExpertLyapunovController
from obstacle_manager import ObstacleManager


NUM_EPISODES = 120
MAX_STEPS_PER_EPISODE = 600

IMG_WIDTH = 320
IMG_HEIGHT = 240
FOV = 90
DT = 1.0 / 60.0

# 2.0x Scaled Environment Parameters
ARENA_HALF_LENGTH = 16.0
ARENA_HALF_WIDTH = 8.0

START_POS = np.array([-12.0, 0.0], dtype=np.float32)
TARGET_POS = np.array([12.0, 0.0], dtype=np.float32)

ROBOT_Z = 0.1

GOAL_THRESHOLD = 0.50
COLLISION_THRESHOLD = 0.64
WALL_THRESHOLD = 0.60


def get_15_obstacle_initial_positions():
    # 2.0x scaled obstacle initial positions
    pursuers = [
        [-7.0,  3.0, 0.4], [-7.0, -3.0, 0.4],
        [-3.6,  1.6, 0.4], [-3.6, -1.6, 0.4],
        [-1.0,  3.6, 0.4]
    ]
    defenders = [
        [9.6,  2.4, 0.4], [9.6, -2.4, 0.4],
        [11.0, 1.0, 0.4], [11.0, -1.0, 0.4],
        [12.4, 3.0, 0.4]
    ]
    roamers = [
        [1.6,  4.4, 0.4], [1.6, -4.4, 0.4],
        [5.0,  3.6, 0.4], [5.0, -3.6, 0.4],
        [7.6,  0.0, 0.4]
    ]
    return pursuers + defenders + roamers


def create_arena_walls(half_length, half_width, wall_height=1.0, wall_thickness=0.2):
    wall_ids = []
    z_center = wall_height / 2.0

    wall_specs = [
        ([half_length, wall_thickness / 2.0, wall_height / 2.0], [0, half_width, z_center]),
        ([half_length, wall_thickness / 2.0, wall_height / 2.0], [0, -half_width, z_center]),
        ([wall_thickness / 2.0, half_width, wall_height / 2.0], [half_length, 0, z_center]),
        ([wall_thickness / 2.0, half_width, wall_height / 2.0], [-half_length, 0, z_center])
    ]

    for half_extents, pos in wall_specs:
        visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=[0.6, 0.6, 0.6, 1])
        collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        wall_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_shape,
            baseCollisionShapeIndex=collision_shape,
            basePosition=pos
        )
        wall_ids.append(wall_id)

    return wall_ids


def create_robot():
    visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.4, 0.4, 0.2], rgbaColor=[0, 0, 1, 1])
    collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.4, 0.4, 0.2])
    robot_id = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=visual_shape,
        baseCollisionShapeIndex=collision_shape,
        basePosition=[START_POS[0], START_POS[1], ROBOT_Z]
    )
    return robot_id


def create_goal():
    visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=0.4, rgbaColor=[0, 1, 0, 1])
    goal_id = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=visual_shape,
        basePosition=[TARGET_POS[0], TARGET_POS[1], 0.0]
    )
    return goal_id


def create_obstacles(initial_positions):
    obstacle_ids = []
    for pos in initial_positions:
        visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.24, 0.24, 0.4], rgbaColor=[1, 0, 0, 1])
        collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.24, 0.24, 0.4])
        obstacle_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_shape,
            baseCollisionShapeIndex=collision_shape,
            basePosition=pos
        )
        obstacle_ids.append(obstacle_id)
    return obstacle_ids


def sync_robot_to_pybullet(robot_id, robot_pos, robot_yaw):
    quat = p.getQuaternionFromEuler([0.0, 0.0, robot_yaw])
    p.resetBasePositionAndOrientation(
        robot_id,
        [float(robot_pos[0]), float(robot_pos[1]), ROBOT_Z],
        quat
    )


def sync_obstacles_to_pybullet(obstacle_ids, obstacle_positions):
    for idx, obstacle_id in enumerate(obstacle_ids):
        p.resetBasePositionAndOrientation(
            obstacle_id,
            obstacle_positions[idx],
            [0, 0, 0, 1]
        )


def check_collision(robot_pos, obstacle_positions):
    robot_xy = np.asarray(robot_pos[:2], dtype=np.float32)
    for obs in obstacle_positions:
        obs_xy = np.asarray(obs[:2], dtype=np.float32)
        if np.linalg.norm(robot_xy - obs_xy) < COLLISION_THRESHOLD:
            return True
    return False


def check_wall_hit(robot_pos):
    x, y = robot_pos[0], robot_pos[1]
    return abs(x) > ARENA_HALF_LENGTH - WALL_THRESHOLD or abs(y) > ARENA_HALF_WIDTH - WALL_THRESHOLD


class PyBulletVisionSystem:
    def __init__(self, robot_id):
        self.robot_id = robot_id
        self.width = IMG_WIDTH
        self.height = IMG_HEIGHT
        self.fov = FOV
        self.extractor = CNNFeatureExtractor(img_width=IMG_WIDTH, img_height=IMG_HEIGHT, fov_deg=FOV, stack_frames=2)

    def reset(self):
        self.extractor.reset()

    def capture_observation(self):
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        rot_matrix = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        cam_pos = np.array(pos) + np.array([0.0, 0.0, 0.35])
        target_cam = cam_pos + rot_matrix.dot(np.array([1.0, 0.0, 0.0]))

        view_matrix = p.computeViewMatrix(cam_pos, target_cam, [0, 0, 1])
        proj_matrix = p.computeProjectionMatrixFOV(self.fov, self.width / self.height, 0.02, 20.0)

        _, _, rgb, _, _ = p.getCameraImage(self.width, self.height, view_matrix, proj_matrix)
        rgba = np.array(rgb, dtype=np.uint8).reshape((self.height, self.width, 4))
        observation = self.extractor.extract_observation(rgba)

        return rgba, observation


def collect_demonstrations():
    print("=" * 70)
    print("  Collecting Scaled Expert Demonstration Dataset for GAIL")
    print("  15 Scaled Dynamic Obstacles | Domain [-16, 16] x [-8, 8]")
    print("  Scaled Kinematic Lyapunov Expert + PyBullet Vision System")
    print("=" * 70)

    observations = []
    actions = []

    goal_reached_count = 0
    collision_count = 0
    wall_hit_count = 0
    timeout_count = 0

    final_distances = []
    episode_lengths = []

    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)

    for episode in range(NUM_EPISODES):
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.loadURDF("plane.urdf")

        create_arena_walls(ARENA_HALF_LENGTH, ARENA_HALF_WIDTH)
        robot_id = create_robot()
        create_goal()

        initial_obs = get_15_obstacle_initial_positions()
        obstacle_ids = create_obstacles(initial_obs)
        obs_mgr = ObstacleManager(
            initial_obs,
            speed=0.40,
            separation_dist=1.30,
            arena_bounds=(ARENA_HALF_LENGTH - 1.0, ARENA_HALF_WIDTH - 1.0),
            target_pos=TARGET_POS
        )

        expert = ExpertLyapunovController(
            Kp=4.0, Kd=1.5, Kb=8.0, K_wall=8.0, K_theta=2.5, max_v=4.0, max_omega=2.5
        )

        vision = PyBulletVisionSystem(robot_id)
        vision.reset()

        robot_pos = START_POS.copy()
        robot_yaw = 0.0
        robot_vel = np.zeros(2, dtype=np.float32)

        for step in range(MAX_STEPS_PER_EPISODE):
            current_obs = obs_mgr.update(robot_pos, step_count=step, dt=DT)
            obs_velocities = obs_mgr.get_velocities()

            sync_robot_to_pybullet(robot_id, robot_pos, robot_yaw)
            sync_obstacles_to_pybullet(obstacle_ids, current_obs)

            _, state = vision.capture_observation()

            v_expert, omega_expert, dist = expert.compute_action(
                robot_pos, robot_vel, robot_yaw, TARGET_POS, current_obs, obs_velocities=obs_velocities
            )

            observations.append(state.copy())
            actions.append([float(v_expert), float(omega_expert)])

            robot_pos = robot_pos + v_expert * np.array([np.cos(robot_yaw), np.sin(robot_yaw)], dtype=np.float32) * DT
            robot_yaw = robot_yaw + omega_expert * DT
            robot_yaw = np.arctan2(np.sin(robot_yaw), np.cos(robot_yaw))
            robot_vel = v_expert * np.array([np.cos(robot_yaw), np.sin(robot_yaw)], dtype=np.float32)

            dist_after = np.linalg.norm(robot_pos - TARGET_POS)

            if dist_after < GOAL_THRESHOLD:
                goal_reached_count += 1
                final_distances.append(dist_after)
                episode_lengths.append(step + 1)
                break

            collision = check_collision(robot_pos, current_obs)
            if collision:
                collision_count += 1
                final_distances.append(dist_after)
                episode_lengths.append(step + 1)
                break

            wall_hit = check_wall_hit(robot_pos)
            if wall_hit:
                wall_hit_count += 1
                final_distances.append(dist_after)
                episode_lengths.append(step + 1)
                break
        else:
            timeout_count += 1
            final_dist = np.linalg.norm(robot_pos - TARGET_POS)
            final_distances.append(final_dist)
            episode_lengths.append(MAX_STEPS_PER_EPISODE)

        if (episode + 1) % 20 == 0:
            print(f"  Collected {episode + 1}/{NUM_EPISODES} episodes | Total Samples: {len(observations)}")

    p.disconnect()

    observations = np.asarray(observations, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)

    np.savez("demo_dataset.npz", observations=observations, actions=actions)

    print()
    print("=" * 70)
    print("  DEMONSTRATION COLLECTION RESULTS")
    print("=" * 70)
    print(f"Total Episodes      : {NUM_EPISODES}")
    print(f"Total Samples       : {len(observations)}")
    print(f"Goal Reached        : {goal_reached_count}/{NUM_EPISODES} ({100.0*goal_reached_count/NUM_EPISODES:.2f}%)")
    print(f"Collisions          : {collision_count}/{NUM_EPISODES} ({100.0*collision_count/NUM_EPISODES:.2f}%)")
    print(f"Wall Hits           : {wall_hit_count}/{NUM_EPISODES} ({100.0*wall_hit_count/NUM_EPISODES:.2f}%)")
    print(f"Timeouts            : {timeout_count}/{NUM_EPISODES} ({100.0*timeout_count/NUM_EPISODES:.2f}%)")
    print(f"Average Final Dist  : {np.mean(final_distances):.3f} m")
    print(f"Average Ep Length   : {np.mean(episode_lengths):.2f} steps")
    print("=" * 70)


if __name__ == "__main__":
    collect_demonstrations()
