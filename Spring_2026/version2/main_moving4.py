import pybullet as p
import pybullet_data
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import os
from feature_extractor import CNNFeatureExtractor



class NumpyNeuralNet:
    def __init__(self, layer_sizes=[16, 128, 64, 2]):
        self.layer_sizes = layer_sizes
        self.weights = [np.zeros((layer_sizes[i], layer_sizes[i+1]))
                        for i in range(len(layer_sizes)-1)]
        self.biases  = [np.zeros((1, layer_sizes[i+1]))
                        for i in range(len(layer_sizes)-1)]

    def relu(self, x):     return np.maximum(0, x)
    def softplus(self, x): return np.where(x>20, x, np.log1p(np.exp(np.clip(x,-500,20))))

    def forward(self, x):
        current = x.reshape(1, -1)
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = current @ W + b
            current = self.relu(z) if i < len(self.weights)-1 else self.softplus(z)+0.5
        return current.flatten()   # [Kp, Kd]

    def load_weights(self, path="nn_weights.npz"):
        data = np.load(path)
        for i in range(len(self.weights)):
            self.weights[i] = data[f"W{i}"]
            self.biases[i]  = data[f"b{i}"]
        print(f"[INFO] Loaded NN weights from '{path}'")




def load_network():
    net = NumpyNeuralNet(layer_sizes=[16, 128, 64, 2])
    if os.path.exists("nn_weights.npz"):
        net.load_weights("nn_weights.npz")
        print("[INFO] Mode: DYNAMIC — live camera → features → Kp, Kd each step")
        return net, None, None
    elif os.path.exists("trained_gains.json"):
        data = json.load(open("trained_gains.json"))
        print(f"[WARN] Using fixed fallback gains: Kp={data['Kp']}, Kd={data['Kd']}")
        return None, data["Kp"], data["Kd"]
    else:
        print("[WARN] No weights found. Using defaults Kp=2.5, Kd=1.2")
        return None, 2.5, 1.2

GAIN_NET, FALLBACK_Kp, FALLBACK_Kd = load_network()
EXTRACTOR = CNNFeatureExtractor(img_width=320, img_height=240, fov_deg=90)




class KalmanFilter:
    def __init__(self, dt, process_noise_std=0.1, meas_pos_std=0.3, meas_vel_std=0.2):
        self.dt = dt
        self.x  = np.zeros((4, 1))
        self.F  = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]])
        self.H  = np.eye(4)
        self.P  = np.eye(4)
        self.Q  = np.eye(4) * process_noise_std**2
        self.R  = np.diag([meas_pos_std**2, meas_pos_std**2,
                           meas_vel_std**2,  meas_vel_std**2])

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, z):
        z = np.array(z).reshape(4,1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.x




class VisionSystem:
    def __init__(self, robot_id):
        self.robot_id = robot_id
        self.width    = 320
        self.height   = 240
        self.fov      = 90

    def capture_and_extract(self):
        pos, orn   = p.getBasePositionAndOrientation(self.robot_id)
        rot_matrix = np.array(p.getMatrixFromQuaternion(orn)).reshape(3,3)
        cam_pos    = np.array(pos) + np.array([0, 0, 0.25])
        target_pos = cam_pos + rot_matrix.dot(np.array([1, 0, 0]))

        view_matrix = p.computeViewMatrix(cam_pos, target_pos, [0,0,1])
        proj_matrix = p.computeProjectionMatrixFOV(
            self.fov, self.width/self.height, 0.02, 10)

        _, _, rgb, _, _ = p.getCameraImage(
            self.width, self.height, view_matrix, proj_matrix)

        rgba = np.array(rgb, dtype=np.uint8).reshape((self.height, self.width, 4))
        features     = EXTRACTOR.extract(rgba)
        vel_estimate = [float(features[14]), float(features[15])]

        return rgba, features, vel_estimate




STOP_DIST    = 0.55   # hard stop / reverse trigger (metres)
CAUTION_DIST = 1.20   # gain modulation begins (metres)

def check_proximity(robot_pos, obstacle_positions):
    min_dist    = float('inf')
    nearest_pos = None
    for obs in obstacle_positions:
        d = np.linalg.norm(np.array(robot_pos[:2]) - np.array(obs[:2]))
        if d < min_dist:
            min_dist    = d
            nearest_pos = obs
    danger_stop      = min_dist <= STOP_DIST
    proximity_factor = np.clip(
        (CAUTION_DIST - min_dist) / (CAUTION_DIST - STOP_DIST), 0.0, 1.0)
    return danger_stop, min_dist, nearest_pos, proximity_factor




KP_SAFE = 0.6
KD_SAFE = 3.5

def modulate_gains(Kp_nn, Kd_nn, proximity_factor):
    Kp = (1.0 - proximity_factor) * Kp_nn + proximity_factor * KP_SAFE
    Kd = (1.0 - proximity_factor) * Kd_nn + proximity_factor * KD_SAFE
    return Kp, Kd



FRONT_HALF_ANGLE = np.deg2rad(60)   # ±60° cone counts as "front"
FRONT_BLOCK_DIST = 1.00             # obstacle must be closer than this

def scan_front_arc(robot_pos, robot_yaw, obstacle_positions):
    robot_fwd = np.array([np.cos(robot_yaw), np.sin(robot_yaw)])
    robot_left = np.array([-np.sin(robot_yaw), np.cos(robot_yaw)])

    closest_front  = float('inf')
    left_clearance = 0.0
    right_clearance= 0.0

    for obs in obstacle_positions:
        vec = np.array(obs[:2]) - np.array(robot_pos[:2])
        d   = np.linalg.norm(vec)
        if d < 1e-3:
            continue
        vec_n = vec / d

        lateral = np.dot(vec_n, robot_left)
        forward = np.dot(vec_n, robot_fwd)

        angle = np.arccos(np.clip(forward, -1, 1))

        if angle < FRONT_HALF_ANGLE and d < FRONT_BLOCK_DIST:
            closest_front = min(closest_front, d)
            if lateral >= 0:
                left_clearance  += 1.0 / (d + 0.1)
            else:
                right_clearance += 1.0 / (d + 0.1)

    front_blocked = closest_front < FRONT_BLOCK_DIST
    # side with LESS accumulated weight = more room to steer toward
    clear_side = -1.0 if left_clearance >= right_clearance else 1.0
    return front_blocked, closest_front, clear_side



def lyapunov_controller(current_pos, current_vel, current_yaw,
                        target_pos, obstacles, Kp, Kd,
                        front_blocked=False, clear_side=1.0,
                        closest_front=float('inf')):
    K_theta     = 2.0
    K_b         = 3.5
    ROBOT_RADIUS= 0.65

    pos_error    = np.array(target_pos[:2]) - np.array(current_pos[:2])
    dist_to_goal = np.linalg.norm(pos_error)

    if dist_to_goal < 0.1:
        return 0.0, 0.0, dist_to_goal, current_yaw

    # ── Attractive term (scaled down when front is blocked) ────────
    att_scale = 0.15 if front_blocked else 1.0
    u_att     = pos_error * Kp * att_scale - np.array(current_vel[:2]) * Kd

    # ── Barrier / repulsion term ────────────────────────────────────
    nearest = sorted(
        [(np.linalg.norm(np.array(current_pos[:2])-np.array(o[:2])), o)
         for o in obstacles], key=lambda x: x[0])[:3]

    u_barrier = np.zeros(2)
    for d_center, obs in nearest:
        obs_vec = np.array(current_pos[:2]) - np.array(obs[:2])
        if d_center < 1e-3:
            continue
        h = d_center - ROBOT_RADIUS
        if h < 1.0:
            h           = max(h, 0.02)
            rep_dir     = obs_vec / d_center
            barrier_mag = K_b / (h + 0.5) ** 2
            t1 = np.array([-rep_dir[1], rep_dir[0]])
            t2 = np.array([ rep_dir[1],-rep_dir[0]])
            tangent    = t1 if np.dot(t1, pos_error) > np.dot(t2, pos_error) else t2
            u_barrier += barrier_mag * rep_dir + 2.0 * barrier_mag * tangent

    u_total = u_att + u_barrier + np.array([0.0, 0.15])

    if front_blocked:
        robot_left = np.array([-np.sin(current_yaw), np.cos(current_yaw)])
        u_total    = u_total + robot_left * clear_side * 2.5

    theta_des = np.arctan2(u_total[1], u_total[0])
    e_theta   = np.arctan2(
        np.sin(theta_des - current_yaw),
        np.cos(theta_des - current_yaw))

    if front_blocked:
        reverse_scale = np.clip(
            (FRONT_BLOCK_DIST - closest_front) / (FRONT_BLOCK_DIST - STOP_DIST),
            0.0, 1.0)
        v = -(0.25 + 0.35 * reverse_scale)   # –0.25 … –0.60 m/s
        print(f"[REVERSE] front_blocked dist={closest_front:.2f} "
              f"v={v:.2f} side={'LEFT' if clear_side>0 else 'RIGHT'}")
    else:
        v = min(np.linalg.norm(u_total), 1.2)

    omega = np.clip(K_theta * e_theta, -2.5, 2.5)
    return v, omega, dist_to_goal, theta_des



def main():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.loadURDF("plane.urdf")

    net = NumpyNeuralNet(layer_sizes=[16, 128, 64, 2])
    net.load_weights("nn_weights2.npz")   # 👈 YOUR NEW FILE
    print("[INFO] Running in IMITATION LEARNING mode")

    robot_id = p.createMultiBody(
        baseMass=1,
        baseVisualShapeIndex=p.createVisualShape(
            p.GEOM_BOX, halfExtents=[0.2,0.2,0.1], rgbaColor=[0,0,1,1]),
        baseCollisionShapeIndex=p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[0.2,0.2,0.1]),
        basePosition=[0,0,0.1]
    )

    target_pos = [6, 0, 0]

    p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=p.createVisualShape(
            p.GEOM_SPHERE, radius=0.2, rgbaColor=[0,1,0,1]),
        basePosition=target_pos
    )

    initial_obs = [
        [2.0,  1.5, 0.5],
        [2.0, -1.5, 0.5],
    
        [3.5,  1.5, 0.5],
        [3.5, -1.5, 0.5],
    
        [5.0,  1.5, 0.5],
        [5.0, -1.5, 0.5],
    ]

    obs_ids = []
    for pos in initial_obs:
        oid = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=p.createVisualShape(
                p.GEOM_BOX, halfExtents=[0.25,0.25,0.5], rgbaColor=[1,0,0,1]),
            baseCollisionShapeIndex=p.createCollisionShape(
                p.GEOM_BOX, halfExtents=[0.25,0.25,0.5]),
            basePosition=pos)
        obs_ids.append(oid)

    kf     = KalmanFilter(dt=1/240.0)
    vision = VisionSystem(robot_id)

    for i in range(5000):

        # Move obstacles
        current_obs_positions = []
        for idx, oid in enumerate(obs_ids):
            base = initial_obs[idx]
            new_x = base[0] + 0.4 * np.sin(i * 0.015 + idx)
            new_y = base[1] + 0.4 * np.cos(i * 0.015 + idx)
            p.resetBasePositionAndOrientation(
                oid, [new_x, new_y, base[2]], [0,0,0,1])
            current_obs_positions.append([new_x, new_y, base[2]])

        p.stepSimulation()

        # robot state
        gt_pos, gt_orn = p.getBasePositionAndOrientation(robot_id)
        gt_vel, _      = p.getBaseVelocity(robot_id)
        _, _, yaw      = p.getEulerFromQuaternion(gt_orn)


        _, features, vel_est = vision.capture_and_extract()

        Kp_nn, Kd_nn = net.forward(features)

        danger_stop, min_dist, nearest_obs, prox = check_proximity(
            gt_pos, current_obs_positions)

        Kp, Kd = modulate_gains(Kp_nn, Kd_nn, prox)

        front_blocked, closest_front, clear_side = scan_front_arc(
            gt_pos, yaw, current_obs_positions)

        v, omega, dist, theta_des = lyapunov_controller(
            [gt_pos[0], gt_pos[1]],
            vel_est,
            yaw,
            target_pos,
            current_obs_positions,
            Kp=Kp,
            Kd=Kd,
            front_blocked=front_blocked,
            clear_side=clear_side,
            closest_front=closest_front
        )

        p.resetBaseVelocity(
            robot_id,
            linearVelocity=[v*np.cos(theta_des), v*np.sin(theta_des), 0],
            angularVelocity=[0, 0, omega]
        )

        p.resetDebugVisualizerCamera(5, 0, -50, gt_pos)

        time.sleep(1/240.0)

        if dist < 0.3:
            print("Target reached using imitation policy!")
            break

    p.disconnect()

if __name__ == "__main__":
    main()