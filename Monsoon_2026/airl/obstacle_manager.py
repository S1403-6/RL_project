"""
obstacle_manager.py (GAIL Scaled Version)
===========================================
Manages 15 scaled dynamic obstacles across 3 behavior modes:
  1. Pursuers (5): Target the robot's position at scaled speed (0.40 m/s).
  2. Goal Defenders (5): Patrol around the scaled target goal (12.0, 0.0).
  3. Random Roamers (5): Move in randomized directions across scaled arena [-16, 16] x [-8, 8].

Obstacle positions and velocities are scaled up by 2.0x relative to Version 3.
"""

import numpy as np

class ObstacleManager:
    def __init__(self, initial_positions, speed=0.40, separation_dist=1.30, arena_bounds=(15.0, 7.5), target_pos=(12.0, 0.0)):
        self.obs_positions = np.array(initial_positions, dtype=np.float32)[:, :2]
        self.speed = speed
        self.separation_dist = separation_dist
        self.arena_max_x, self.arena_max_y = arena_bounds
        self.target_pos = np.array(target_pos, dtype=np.float32)
        self.num_obs = len(self.obs_positions)
        self.velocities = np.zeros_like(self.obs_positions)

        # Behaviors across 15 obstacles:
        # First 5: Pursuers, Next 5: Goal Defenders, Last 5: Random Roamers
        self.behaviors = []
        for i in range(self.num_obs):
            if i < 5:
                self.behaviors.append("pursuit")
            elif i < 10:
                self.behaviors.append("goal_cluster")
            else:
                self.behaviors.append("random_roam")

        np.random.seed(42)
        random_angles = np.random.uniform(0, 2 * np.pi, size=self.num_obs)
        self.random_dirs = np.column_stack([np.cos(random_angles), np.sin(random_angles)])

    def reset(self, initial_positions):
        self.obs_positions = np.array(initial_positions, dtype=np.float32)[:, :2]
        self.velocities = np.zeros_like(self.obs_positions)

    def update(self, robot_pos, step_count=0, dt=1/60.0):
        """
        Updates obstacle positions:
          - Pursuers target robot
          - Goal defenders circle goal (12.0, 0.0)
          - Random roamers move along random directions
          - Mutual separation repels obstacles from each other
          - Arena boundary bouncing
        """
        robot_p = np.array(robot_pos[:2], dtype=np.float32)

        for i in range(self.num_obs):
            pos_i = self.obs_positions[i]
            mode = self.behaviors[i]

            # 1. Behavior Force Vector
            if mode == "pursuit":
                vec_to_robot = robot_p - pos_i
                dist_r = np.linalg.norm(vec_to_robot)
                u_behavior = vec_to_robot / (dist_r + 1e-5) if dist_r > 0.2 else np.zeros(2)

            elif mode == "goal_cluster":
                goal_vec = self.target_pos - pos_i
                dist_g = np.linalg.norm(goal_vec)
                tangent = np.array([-goal_vec[1], goal_vec[0]])
                if dist_g > 3.6:
                    u_behavior = 0.5 * (goal_vec / (dist_g + 1e-5)) + 0.8 * (tangent / (dist_g + 1e-5))
                else:
                    u_behavior = tangent / (dist_g + 1e-5)

            else:  # "random_roam"
                u_behavior = self.random_dirs[i] + 0.3 * np.array([
                    np.sin(step_count * 0.05 + i),
                    np.cos(step_count * 0.05 + i * 0.7)
                ])

            norm_b = np.linalg.norm(u_behavior)
            if norm_b > 1e-4:
                u_behavior = u_behavior / norm_b

            # 2. Mutual Separation Force
            u_separation = np.zeros(2)
            for j in range(self.num_obs):
                if i == j:
                    continue
                pos_j = self.obs_positions[j]
                vec_ij = pos_i - pos_j
                dist_ij = np.linalg.norm(vec_ij)
                if dist_ij < self.separation_dist and dist_ij > 1e-4:
                    u_separation += (vec_ij / dist_ij) * (self.separation_dist - dist_ij) / (dist_ij + 0.10)

            # 3. Combine behavior + mutual separation
            dir_total = 0.6 * u_behavior + 1.4 * u_separation
            norm_dir = np.linalg.norm(dir_total)
            if norm_dir > 1e-4:
                desired_vel = (dir_total / norm_dir) * self.speed
            else:
                desired_vel = np.zeros(2)

            self.velocities[i] = 0.82 * self.velocities[i] + 0.18 * desired_vel
            new_pos = pos_i + self.velocities[i] * dt

            # 4. Arena Boundary Bouncing
            if abs(new_pos[0]) > self.arena_max_x - 0.6:
                self.velocities[i, 0] *= -0.8
                self.random_dirs[i, 0] *= -1.0
                new_pos[0] = np.clip(new_pos[0], -self.arena_max_x + 0.6, self.arena_max_x - 0.6)
            if abs(new_pos[1]) > self.arena_max_y - 0.6:
                self.velocities[i, 1] *= -0.8
                self.random_dirs[i, 1] *= -1.0
                new_pos[1] = np.clip(new_pos[1], -self.arena_max_y + 0.6, self.arena_max_y - 0.6)

            self.obs_positions[i] = new_pos

        return [[pos[0], pos[1], 0.5] for pos in self.obs_positions]

    def get_velocities(self):
        return self.velocities.copy()
