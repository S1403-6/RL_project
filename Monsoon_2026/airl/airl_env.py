import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data

from gymnasium import spaces

from feature_extractor import CNNFeatureExtractor
from obstacle_manager import ObstacleManager


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

MAX_STEPS = 800
DT = 1.0 / 60.0

MAX_LINEAR_VELOCITY = 4.0
MAX_ANGULAR_VELOCITY = 2.5

OBSTACLE_SPEED = 0.40
OBSTACLE_SEPARATION = 1.30

PROGRESS_REWARD_SCALE = 5.0
GOAL_REWARD = 100.0
COLLISION_PENALTY = 50.0
WALL_PENALTY = 50.0
TIME_PENALTY = 0.01

# Dense obstacle-safety shaping.
# Clearance is measured from the robot/obstacle collision boundary.
CLEARANCE_SAFE_DISTANCE = 2.5
CLEARANCE_PENALTY_DISTANCE = 2.0
CLEARANCE_REWARD_SCALE = 0.20
OBSTACLE_PENALTY_SCALE = 3.0

CAM_WIDTH = 320
CAM_HEIGHT = 240


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


class AirlNavEnv(gym.Env):
    """
    AIRL navigation environment.

    This contains the full navigation environment (previously defined
    as GailNavEnv in gail_env.py) so that AIRL no longer depends on an
    external import. The environment logic itself is unchanged from
    the GAIL version so that GAIL-PPO and AIRL-PPO comparisons stay
    fair.

    The only thing that differs between the AIRL and GAIL setups is
    the reward supplied to PPO, which is handled by
    AIRLRewardWrapper below.
    """

    metadata = {
        "render_modes": ["human"]
    }

    def __init__(
        self,
        render_gui=False,
        normalize_obs=True
    ):

        super().__init__()

        self.render_gui = render_gui
        self.normalize_obs = normalize_obs

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(32,),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )

        self.physics_client = p.connect(
            p.GUI if render_gui else p.DIRECT
        )

        p.setAdditionalSearchPath(
            pybullet_data.getDataPath()
        )

        p.setGravity(
            0,
            0,
            -9.8
        )

        self.robot_id = None
        self.goal_id = None

        self.obstacle_ids = []

        self.obs_mgr = None

        self.robot_pos = START_POS.copy()
        self.robot_yaw = 0.0

        self.step_count = 0
        self.previous_distance = None

        self.extractor = CNNFeatureExtractor(
            img_width=CAM_WIDTH,
            img_height=CAM_HEIGHT,
            fov_deg=90,
            stack_frames=2
        )

        self.obs_mean = None
        self.obs_std = None

        self._load_observation_normalization()

    def _load_observation_normalization(self):

        try:
            data = np.load(
                "imitation_policy_weights.npz"
            )
        except FileNotFoundError as exc:
            if self.normalize_obs:
                raise FileNotFoundError(
                    "imitation_policy_weights.npz is required when "
                    "normalize_obs=True."
                ) from exc

            self.obs_mean = None
            self.obs_std = None
            return

        if "mean" not in data or "std" not in data:
            if self.normalize_obs:
                raise ValueError(
                    "imitation_policy_weights.npz must contain both "
                    "'mean' and 'std' for normalized observations."
                )

            self.obs_mean = None
            self.obs_std = None
            return

        self.obs_mean = data["mean"].astype(np.float32)
        self.obs_std = data["std"].astype(np.float32) + 1e-6

        if self.obs_mean.shape != (32,) or self.obs_std.shape != (32,):
            raise ValueError(
                "Observation normalization mean/std must both "
                "have shape (32,)."
            )

    def reset(
        self,
        seed=None,
        options=None
    ):

        super().reset(seed=seed)

        p.resetSimulation()

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

        self._create_arena_walls()

        self._create_robot()

        self._create_goal()

        initial_obstacles = (
            get_15_obstacle_initial_positions()
        )

        self._create_obstacles(
            initial_obstacles
        )

        self.obs_mgr = ObstacleManager(
            initial_obstacles,
            speed=OBSTACLE_SPEED,
            separation_dist=OBSTACLE_SEPARATION,
            arena_bounds=(
                ARENA_HALF_LENGTH - 1.0,
                ARENA_HALF_WIDTH - 1.0
            ),
            target_pos=TARGET_POS
        )

        self.robot_pos = START_POS.copy()
        self.robot_yaw = 0.0

        self.step_count = 0

        self.previous_distance = np.linalg.norm(
            self.robot_pos - TARGET_POS
        )

        self.extractor.reset()

        observation = self._get_observation()

        info = {
            "robot_pos": self.robot_pos.copy(),
            "distance_to_goal": float(
                self.previous_distance
            )
        }

        return observation, info

    def _create_robot(self):

        visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[
                0.4,
                0.4,
                0.2
            ],
            rgbaColor=[
                0,
                0,
                1,
                1
            ]
        )

        collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[
                0.4,
                0.4,
                0.2
            ]
        )

        self.robot_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual,
            baseCollisionShapeIndex=collision,
            basePosition=[
                START_POS[0],
                START_POS[1],
                ROBOT_Z
            ]
        )

    def _create_goal(self):

        visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.4,
            rgbaColor=[
                0,
                1,
                0,
                1
            ]
        )

        self.goal_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual,
            basePosition=[
                TARGET_POS[0],
                TARGET_POS[1],
                0.0
            ]
        )

    def _create_obstacles(
        self,
        positions
    ):

        self.obstacle_ids = []

        for position in positions:

            visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[
                    0.24,
                    0.24,
                    0.4
                ],
                rgbaColor=[
                    1,
                    0,
                    0,
                    1
                ]
            )

            collision = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[
                    0.24,
                    0.24,
                    0.4
                ]
            )

            obstacle_id = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=visual,
                baseCollisionShapeIndex=collision,
                basePosition=position
            )

            self.obstacle_ids.append(
                obstacle_id
            )

    def normalized_to_physical_action(
        self,
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

        v_norm = float(action[0])
        omega_norm = float(action[1])

        v = (
            0.5
            * (v_norm + 1.0)
            * MAX_LINEAR_VELOCITY
        )

        omega = (
            omega_norm
            * MAX_ANGULAR_VELOCITY
        )

        return v, omega

    def step(
        self,
        action
    ):

        self.step_count += 1

        action = np.asarray(
            action,
            dtype=np.float32
        )

        action = np.clip(
            action,
            -1.0,
            1.0
        )

        v, omega = (
            self.normalized_to_physical_action(
                action
            )
        )

        current_obstacles = (
            self.obs_mgr.update(
                self.robot_pos,
                step_count=self.step_count,
                dt=DT
            )
        )

        direction = np.array(
            [
                np.cos(self.robot_yaw),
                np.sin(self.robot_yaw)
            ],
            dtype=np.float32
        )

        self.robot_pos += (
            v
            * direction
            * DT
        )

        self.robot_yaw += (
            omega * DT
        )

        self.robot_yaw = np.arctan2(
            np.sin(self.robot_yaw),
            np.cos(self.robot_yaw)
        )

        self._update_robot_in_pybullet()

        self._update_obstacles_in_pybullet(
            current_obstacles
        )

        p.stepSimulation()

        observation = self._get_observation()

        distance = np.linalg.norm(
            self.robot_pos - TARGET_POS
        )

        goal_reached = (
            distance < GOAL_THRESHOLD
        )

        collision = (
            self._check_collision(
                current_obstacles
            )
        )

        min_obstacle_distance = self._get_min_obstacle_distance(
            current_obstacles
        )

        obstacle_clearance = max(
            0.0,
            min_obstacle_distance - COLLISION_THRESHOLD
        )

        clearance_reward = CLEARANCE_REWARD_SCALE * np.clip(
            obstacle_clearance / CLEARANCE_SAFE_DISTANCE,
            0.0,
            1.0
        )

        obstacle_penalty = 0.0
        if obstacle_clearance < CLEARANCE_PENALTY_DISTANCE:
            danger = 1.0 - np.clip(
                obstacle_clearance / CLEARANCE_PENALTY_DISTANCE,
                0.0,
                1.0
            )
            obstacle_penalty = OBSTACLE_PENALTY_SCALE * (danger ** 2)

        wall_hit = (
            self._check_wall_hit()
        )

        terminated = (
            goal_reached
            or collision
            or wall_hit
        )

        truncated = (
            self.step_count >= MAX_STEPS
        )

        progress = (
            self.previous_distance
            - distance
        )

        self.previous_distance = distance

        progress_reward = (
            PROGRESS_REWARD_SCALE
            * progress
        )

        reward = (
            progress_reward
            + clearance_reward
            - obstacle_penalty
            - TIME_PENALTY
        )

        if goal_reached:

            reward += GOAL_REWARD

        if collision:

            reward -= COLLISION_PENALTY

        if wall_hit:

            reward -= WALL_PENALTY

        info = {
            "goal_reached": bool(
                goal_reached
            ),
            "collision": bool(
                collision
            ),
            "wall_hit": bool(
                wall_hit
            ),
            "dist": float(
                distance
            ),
            "progress": float(
                progress
            ),
            "progress_reward": float(
                progress_reward
            ),
            "min_obstacle_distance": float(
                min_obstacle_distance
            ),
            "obstacle_clearance": float(
                obstacle_clearance
            ),
            "clearance_reward": float(
                clearance_reward
            ),
            "obstacle_penalty": float(
                obstacle_penalty
            ),
            "safety_reward": float(
                clearance_reward - obstacle_penalty
            ),
            "robot_pos": self.robot_pos.copy(),
            "robot_yaw": float(
                self.robot_yaw
            ),
            "v": float(v),
            "omega": float(omega)
        }

        return (
            observation,
            float(reward),
            terminated,
            truncated,
            info
        )

    def _update_robot_in_pybullet(self):

        quaternion = p.getQuaternionFromEuler(
            [
                0.0,
                0.0,
                self.robot_yaw
            ]
        )

        p.resetBasePositionAndOrientation(
            self.robot_id,
            [
                float(self.robot_pos[0]),
                float(self.robot_pos[1]),
                ROBOT_Z
            ],
            quaternion
        )

    def _update_obstacles_in_pybullet(
        self,
        obstacle_positions
    ):

        for index, obstacle_id in enumerate(
            self.obstacle_ids
        ):

            p.resetBasePositionAndOrientation(
                obstacle_id,
                obstacle_positions[index],
                [
                    0,
                    0,
                    0,
                    1
                ]
            )

    def _get_observation(self):

        position, orientation = (
            p.getBasePositionAndOrientation(
                self.robot_id
            )
        )

        rotation_matrix = np.array(
            p.getMatrixFromQuaternion(
                orientation
            )
        ).reshape(
            3,
            3
        )

        camera_position = (
            np.array(position)
            + np.array(
                [0.0, 0.0, 0.35]
            )
        )

        forward_direction = (
            rotation_matrix.dot(
                np.array(
                    [1.0, 0.0, 0.0]
                )
            )
        )

        camera_target = (
            camera_position
            + forward_direction
        )

        view_matrix = p.computeViewMatrix(
            camera_position,
            camera_target,
            [0, 0, 1]
        )

        projection_matrix = (
            p.computeProjectionMatrixFOV(
                90.0,
                CAM_WIDTH / CAM_HEIGHT,
                0.02,
                20.0
            )
        )

        image = p.getCameraImage(
            CAM_WIDTH,
            CAM_HEIGHT,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix
        )

        rgba = np.asarray(
            image[2],
            dtype=np.uint8
        ).reshape(
            CAM_HEIGHT,
            CAM_WIDTH,
            4
        )

        observation = (
            self.extractor.extract_observation(
                rgba
            )
        )

        observation = np.asarray(
            observation,
            dtype=np.float32
        )

        if (
            self.normalize_obs
            and self.obs_mean is not None
            and self.obs_std is not None
        ):

            observation = (
                observation - self.obs_mean
            ) / self.obs_std

        return observation.astype(
            np.float32
        )

    def _get_min_obstacle_distance(
        self,
        obstacle_positions
    ):
        if obstacle_positions is None or len(obstacle_positions) == 0:
            return float("inf")

        robot_xy = np.asarray(
            self.robot_pos,
            dtype=np.float32
        )

        distances = [
            np.linalg.norm(
                robot_xy - np.asarray(obstacle[:2], dtype=np.float32)
            )
            for obstacle in obstacle_positions
        ]

        return float(np.min(distances))


    def _check_collision(
        self,
        obstacle_positions
    ):

        for obstacle in obstacle_positions:

            obstacle_xy = np.asarray(
                obstacle[:2],
                dtype=np.float32
            )

            distance = np.linalg.norm(
                self.robot_pos
                - obstacle_xy
            )

            if distance < COLLISION_THRESHOLD:

                return True

        return False

    def _check_wall_hit(self):

        x = float(self.robot_pos[0])
        y = float(self.robot_pos[1])

        return (
            abs(x)
            > ARENA_HALF_LENGTH
            - WALL_THRESHOLD
            or
            abs(y)
            > ARENA_HALF_WIDTH
            - WALL_THRESHOLD
        )

    def _create_arena_walls(
        self,
        wall_height=1.0,
        wall_thickness=0.2
    ):

        z_center = (
            wall_height / 2.0
        )

        wall_specs = [

            (
                [
                    ARENA_HALF_LENGTH,
                    wall_thickness / 2.0,
                    wall_height / 2.0
                ],
                [
                    0,
                    ARENA_HALF_WIDTH,
                    z_center
                ]
            ),

            (
                [
                    ARENA_HALF_LENGTH,
                    wall_thickness / 2.0,
                    wall_height / 2.0
                ],
                [
                    0,
                    -ARENA_HALF_WIDTH,
                    z_center
                ]
            ),

            (
                [
                    wall_thickness / 2.0,
                    ARENA_HALF_WIDTH,
                    wall_height / 2.0
                ],
                [
                    ARENA_HALF_LENGTH,
                    0,
                    z_center
                ]
            ),

            (
                [
                    wall_thickness / 2.0,
                    ARENA_HALF_WIDTH,
                    wall_height / 2.0
                ],
                [
                    -ARENA_HALF_LENGTH,
                    0,
                    z_center
                ]
            )
        ]

        for half_extents, position in wall_specs:

            visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=half_extents,
                rgbaColor=[
                    0.6,
                    0.6,
                    0.6,
                    1
                ]
            )

            collision = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=half_extents
            )

            p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=visual,
                baseCollisionShapeIndex=collision,
                basePosition=position
            )

    def close(self):

        if (
            self.physics_client is not None
            and
            p.isConnected(
                self.physics_client
            )
        ):

            p.disconnect(
                self.physics_client
            )

            self.physics_client = None


class AIRLRewardWrapper(gym.Wrapper):
    """
    Combines dense environment safety/progress reward with learned AIRL reward.

    PPO reward:
        R = env_reward + airl_weight * clip(scale * R_AIRL)

    The underlying environment reward is always available in info["env_reward"].
    """

    def __init__(
        self,
        env,
        discriminator=None,
        use_airl_reward=False,
        reward_scale=1.0,
        reward_clip=20.0,
        airl_weight=1.0,
        env_weight=0.05
    ):
        super().__init__(env)
        self.discriminator = discriminator
        self.use_airl_reward = use_airl_reward
        self.reward_scale = float(reward_scale)
        self.reward_clip = float(reward_clip)
        self.airl_weight = float(airl_weight)
        self.env_weight = float(env_weight)
        self.last_obs = None

    def set_discriminator(
        self,
        discriminator,
        use_airl_reward=True
    ):
        self.discriminator = discriminator
        self.use_airl_reward = use_airl_reward

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_obs = np.asarray(obs, dtype=np.float32).copy()
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        if self.last_obs is None:
            raise RuntimeError(
                "AIRLRewardWrapper.step() called before reset()."
            )

        state = self.last_obs.copy()
        next_obs, env_reward, terminated, truncated, info = self.env.step(action)
        next_state = np.asarray(next_obs, dtype=np.float32).copy()
        done = terminated or truncated

        raw_airl_reward = 0.0
        scaled_airl_reward = 0.0
        clipped_airl_reward = 0.0
        weighted_airl_reward = 0.0

        if self.use_airl_reward and self.discriminator is not None:
            raw_airl_reward = float(
                self.discriminator.predict_reward(
                    state,
                    action,
                    next_state,
                    done
                )
            )
            scaled_airl_reward = raw_airl_reward * self.reward_scale
            clipped_airl_reward = float(
                np.clip(
                    scaled_airl_reward,
                    -self.reward_clip,
                    self.reward_clip
                )
            )
            weighted_airl_reward = self.airl_weight * clipped_airl_reward
        
        w_env_reward = self.env_weight * float(env_reward) 

        total_reward =  w_env_reward + weighted_airl_reward

        self.last_obs = next_state

        info = dict(info)
        info["airl_raw_reward"] = float(raw_airl_reward)
        info["airl_scaled_reward"] = float(scaled_airl_reward)
        info["airl_reward"] = float(clipped_airl_reward)
        info["airl_weighted_reward"] = float(weighted_airl_reward)
        info["env_reward"] = float(env_reward)
        info["total_reward"] = float(total_reward)
        info["env_weighted_reward"] = float(w_env_reward)

        return next_obs, total_reward, terminated, truncated, info

