

import numpy as np

class ExpertLyapunovController:
    def __init__(self, Kp=4.0, Kd=1.5, Kb=8.0, K_wall=8.0, K_theta=2.5, max_v=4.0, max_omega=2.5):
        self.Kp = Kp
        self.Kd = Kd
        self.Kb = Kb
        self.K_wall = K_wall
        self.K_theta = K_theta
        self.max_v = max_v
        self.max_omega = max_omega

    def compute_action(self, current_pos, current_vel, current_yaw, target_pos, obstacles, obs_velocities=None, arena_bounds=(15.6, 7.6)):
        """
        Computes scaled expert action (v_E, omega_E).
        """
        pos_error = np.array(target_pos[:2]) - np.array(current_pos[:2])
        dist_to_goal = np.linalg.norm(pos_error)

        if dist_to_goal < 0.2:
            return 0.0, 0.0, dist_to_goal

        goal_dir = pos_error / (dist_to_goal + 1e-5)

        # 1. Goal Attraction
        u_att = pos_error * self.Kp - np.array(current_vel[:2]) * self.Kd

        # 2. Dynamic Obstacle Repulsion & Lateral Dodging
        u_barrier = np.zeros(2)
        r_pos = np.array(current_pos[:2])

        nearest_obs = sorted(
            range(len(obstacles)),
            key=lambda idx: np.linalg.norm(r_pos - np.array(obstacles[idx][:2]))
        )[:15]

        for idx in nearest_obs:
            obs = obstacles[idx]
            obs_pos = np.array(obs[:2])
            obs_vec = r_pos - obs_pos
            d_center = np.linalg.norm(obs_vec)
            if d_center < 1e-3:
                continue

            rep_dir = obs_vec / d_center

            if obs_velocities is not None and idx < len(obs_velocities):
                v_o = np.array(obs_velocities[idx][:2])
            else:
                v_o = -rep_dir * 0.40

            v_closing = max(0.0, np.dot(v_o - np.array(current_vel[:2]), rep_dir))

            clearance = d_center - 0.90
            if clearance < 6.0:
                h = max(clearance, 0.04)
                dynamic_weight = 1.0 + 3.5 * v_closing
                barrier_mag = (self.Kb * dynamic_weight) / ((h + 0.50) ** 2)

                t1 = np.array([-rep_dir[1], rep_dir[0]])
                t2 = np.array([rep_dir[1], -rep_dir[0]])
                tangent = t1 if np.dot(t1, goal_dir) >= np.dot(t2, goal_dir) else t2

                u_barrier += 1.2 * barrier_mag * rep_dir + 2.5 * barrier_mag * tangent

        # 3. Scaled Arena Wall Repulsion
        u_wall = np.zeros(2)
        rx, ry = current_pos[0], current_pos[1]
        max_x, max_y = arena_bounds

        dist_left = rx - (-max_x)
        dist_right = max_x - rx
        dist_bottom = ry - (-max_y)
        dist_top = max_y - ry

        if dist_left < 2.8:
            u_wall[0] += self.K_wall / max(dist_left, 0.1) ** 2
        if dist_right < 2.8:
            u_wall[0] -= self.K_wall / max(dist_right, 0.1) ** 2
        if dist_bottom < 2.8:
            u_wall[1] += self.K_wall / max(dist_bottom, 0.1) ** 2
        if dist_top < 2.8:
            u_wall[1] -= self.K_wall / max(dist_top, 0.1) ** 2

        u_total = u_att + u_barrier + u_wall + np.array([0.0, 0.10])
        theta_des = np.arctan2(u_total[1], u_total[0])
        e_theta = np.arctan2(np.sin(theta_des - current_yaw), np.cos(theta_des - current_yaw))

        v_nominal = np.clip(np.linalg.norm(u_total), 0.0, self.max_v)

        min_clearance = np.inf
        for idx in nearest_obs:
            obs_pos = np.array(obstacles[idx][:2])
            d = np.linalg.norm(r_pos - obs_pos)
            min_clearance = min(min_clearance, d)

        if min_clearance < 2.0:
            speed_scale = np.clip((min_clearance - 0.50) / 1.50, 0.2, 1.0)
        else:
            speed_scale = 1.0

        v_E = v_nominal * speed_scale
        omega_E = np.clip(self.K_theta * e_theta, -self.max_omega, self.max_omega)

        return float(v_E), float(omega_E), float(dist_to_goal)
