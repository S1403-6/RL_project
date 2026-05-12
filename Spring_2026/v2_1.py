import pybullet as p
import pybullet_data
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt

# ==========================================================
# KALMAN FILTER
# ==========================================================

class KalmanFilter:
    def __init__(self, dt, process_noise_std=0.1, meas_pos_std=0.3, meas_vel_std=0.2):
        self.dt = dt
        self.x = np.zeros((4, 1))

        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        self.H = np.eye(4)
        self.P = np.eye(4)
        self.Q = np.eye(4) * process_noise_std**2
        self.R = np.diag([meas_pos_std**2,
                          meas_pos_std**2,
                          meas_vel_std**2,
                          meas_vel_std**2])

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, z):
        z = np.array(z).reshape(4, 1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P
        return self.x


# ==========================================================
# VISION SYSTEM
# ==========================================================

class VisionSystem:
    def __init__(self, robot_id):
        self.robot_id = robot_id
        self.width = 320
        self.height = 240
        self.fov = 90   # <-- Wider FOV for better awareness
        self.focal_length_px = (self.width / 2) / np.tan(np.deg2rad(self.fov / 2))
        self.prev_gray = None

    def capture_camera(self):
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        rot_matrix = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)

        cam_pos = np.array(pos) + np.array([0, 0, 0.25])   # slightly lower camera
        target_pos = cam_pos + rot_matrix.dot(np.array([1, 0, 0]))

        view_matrix = p.computeViewMatrix(cam_pos, target_pos, [0, 0, 1])
        proj_matrix = p.computeProjectionMatrixFOV(self.fov,
                                                   self.width/self.height,
                                                   0.02, 10)

        _, _, rgb, _, _ = p.getCameraImage(self.width, self.height,
                                           view_matrix, proj_matrix)

        rgb_array = np.array(rgb, dtype=np.uint8).reshape(
            (self.height, self.width, 4)
        )
        gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGBA2GRAY)
        return rgb_array, gray



def lyapunov_controller(current_pos, current_vel, current_yaw,
                        target_pos, obstacles):

    Kp = 2.5
    Kd = 1.2
    K_theta = 2.0
    K_b = 2.0        # reduced barrier gain

    pos_error = np.array(target_pos[:2]) - np.array(current_pos[:2])
    dist_to_goal = np.linalg.norm(pos_error)

    if dist_to_goal < 0.1:
        return 0.0, 0.0, dist_to_goal, current_yaw

    # Attractive term
    u_att = (pos_error * Kp) - (np.array(current_vel[:2]) * Kd)

    # --------- ONLY USE 3 NEAREST OBSTACLES ----------
    obs_dists = []
    for obs in obstacles:
        d = np.linalg.norm(np.array(current_pos[:2]) - np.array(obs[:2]))
        obs_dists.append((d, obs))

    obs_dists.sort(key=lambda x: x[0])
    nearest_obs = [o for _, o in obs_dists[:3]]  # only closest 3

    u_barrier = np.array([0.0, 0.0])

    robot_radius = 0.25
    obs_radius = 0.25
    safety_margin = 0.25
    min_dist = robot_radius + obs_radius

    for obs in nearest_obs:
        obs_vec = np.array(current_pos[:2]) - np.array(obs[:2])
        d_center = np.linalg.norm(obs_vec)

        if d_center < 1e-3:
            continue

        d_surface = d_center - min_dist
        h = d_surface - safety_margin

        if h < 1.0:
            if h <= 0.02:
                h = 0.02

            rep_dir = obs_vec / d_center

            barrier_mag = K_b / (h**2)

            t1 = np.array([-rep_dir[1], rep_dir[0]])
            t2 = np.array([ rep_dir[1], -rep_dir[0]])

            tangent = t1 if np.dot(t1, pos_error) > np.dot(t2, pos_error) else t2

            u_barrier += barrier_mag * rep_dir + 2.0 * barrier_mag * tangent

    # combine forces
    u_total = u_att + u_barrier

    # small right-bias to use free space
    u_total += np.array([0.0, 0.15])

    theta_des = np.arctan2(u_total[1], u_total[0])

    e_theta = np.arctan2(
        np.sin(theta_des - current_yaw),
        np.cos(theta_des - current_yaw)
    )

    v = np.linalg.norm(u_total)
    v = min(v, 1.2)

    omega = K_theta * e_theta
    omega = np.clip(omega, -2.0, 2.0)

    return v, omega, dist_to_goal, theta_des



def main():

    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.loadURDF("plane.urdf")

    start_pos = [0, 0, 0.1]
    robot_id = p.createMultiBody(
        baseMass=1,
        baseVisualShapeIndex=p.createVisualShape(
            p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.1],
            rgbaColor=[0, 0, 1, 1]),
        baseCollisionShapeIndex=p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.1]),
        basePosition=start_pos
    )

    target_pos = [6, 0, 0]
    p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=p.createVisualShape(
            p.GEOM_SPHERE, radius=0.2, rgbaColor=[0, 1, 0, 1]),
        basePosition=target_pos
    )

    # ---- You can ADD MORE obstacles here safely ----
    initial_obs_positions = [
        [1.8,  1.2,  0.5],
        [1.8, -1.2,  0.5],
        [2.8,  0.6,  0.5],
        [2.8, -0.6,  0.5],
        [3.6,  1.4,  0.5],
        [3.6, -1.4,  0.5],
        # [4.4,  0.6,  0.5],
        # [4.4, -0.6,  0.5],
        # [5.0,  0.0,  0.5]
    ]

    obs_ids = []
    for pos in initial_obs_positions:
        oid = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=p.createVisualShape(
                p.GEOM_BOX, halfExtents=[0.25, 0.25, 0.5],
                rgbaColor=[1, 0, 0, 1]),
            baseCollisionShapeIndex=p.createCollisionShape(
                p.GEOM_BOX, halfExtents=[0.25, 0.25, 0.5]),
            basePosition=pos
        )
        obs_ids.append(oid)

    kf = KalmanFilter(dt=1/240.0)
    vision = VisionSystem(robot_id)

    V_history = []
    history = {'gt_x': [], 'gt_y': [], 'est_x': [], 'est_y': []}

    for i in range(5000):

        current_obs_positions = []

        for idx, oid in enumerate(obs_ids):
            base_pos = initial_obs_positions[idx]

            amplitude = 0.5
            freq = 0.002
            offset = idx * 0.7

            new_y = base_pos[1] + amplitude * np.sin(i * freq + offset)

            p.resetBasePositionAndOrientation(
                oid,
                [base_pos[0], new_y, base_pos[2]],
                [0, 0, 0, 1]
            )

            current_obs_positions.append(
                [base_pos[0], new_y, base_pos[2]]
            )

        p.stepSimulation()

        gt_pos, gt_orn = p.getBasePositionAndOrientation(robot_id)
        gt_vel, _ = p.getBaseVelocity(robot_id)
        roll, pitch, yaw = p.getEulerFromQuaternion(gt_orn)

        meas_pos_x = gt_pos[0] + np.random.normal(0, 0.15)
        meas_pos_y = gt_pos[1] + np.random.normal(0, 0.15)

        vel_x_vis, vel_y_vis = 0, 0
        if i % 5 == 0:
            _, _ = vision.capture_camera()
            vel_x_vis = gt_vel[0] + np.random.normal(0, 0.1)
            vel_y_vis = gt_vel[1] + np.random.normal(0, 0.1)

        kf.predict()
        state_est = kf.update(
            [meas_pos_x, meas_pos_y, vel_x_vis, vel_y_vis]
        )

        est_pos = [state_est[0][0], state_est[1][0]]
        est_vel = [state_est[2][0], state_est[3][0]]

        pos_err = np.array(est_pos) - np.array(target_pos[:2])
        vel = np.array(est_vel)
        V = 0.5*np.dot(pos_err, pos_err) + 0.5*np.dot(vel, vel)
        V_history.append(V)

        # --- Use GROUND TRUTH position for safer navigation ---
        v, omega, dist, theta_des = lyapunov_controller(
            [gt_pos[0], gt_pos[1]],   # safer than estimate
            est_vel,
            yaw,
            target_pos,
            current_obs_positions
        )

        p.resetBaseVelocity(
            robot_id,
            linearVelocity=[v*np.cos(theta_des), v*np.sin(theta_des), 0],
            angularVelocity=[0, 0, omega]
        )

        p.resetDebugVisualizerCamera(5, 0, -50, gt_pos)

        history['gt_x'].append(gt_pos[0])
        history['gt_y'].append(gt_pos[1])
        history['est_x'].append(est_pos[0])
        history['est_y'].append(est_pos[1])

        time.sleep(1/240.0)

        if dist < 0.2:
            print("Target Reached")
            break

    p.disconnect()

    plt.figure(figsize=(10,5))
    plt.plot(history['gt_x'], history['gt_y'], 'k-', alpha=0.5, label='Ground Truth')
    plt.plot(history['est_x'], history['est_y'], 'b--', label='Kalman Estimate')
    plt.legend()
    plt.title("Navigation Among Moving Obstacles")
    plt.show()

    plt.figure(figsize=(8,4))
    plt.plot(V_history, linewidth=2)
    plt.xlabel("Time step")
    plt.ylabel("V(x, v)")
    plt.title("Lyapunov Energy")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
