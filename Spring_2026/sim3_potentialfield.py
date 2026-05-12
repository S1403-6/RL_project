import pybullet as p
import pybullet_data
import numpy as np
import time
import random
import matplotlib.pyplot as plt

# -------------------- SIMULATION SETUP --------------------
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

SENSOR_STD = 0.05

# -------------------- PARAMETERS (STABILITY CRITICAL) --------------------
K_ATT = 3.0
K_REP = 80.0
D_BASE = 2.0
D_OBS = 6.0

D_SAFE = 2.0        # start repulsion early
D_MIN = 0.6         # prevent singularity
F_REP_MAX = 60.0    # force clamp

def create_body(pos, size, color, mass=0):
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=size)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=size, rgbaColor=color)
    return p.createMultiBody(baseMass=mass,
                             baseCollisionShapeIndex=col,
                             baseVisualShapeIndex=vis,
                             basePosition=pos)

def get_random_positions(n, min_dist=5.0):
    positions = []
    while len(positions) < n:
        test_pos = np.array([random.uniform(2, 13), random.uniform(2, 13), 0])
        if np.linalg.norm(test_pos[:2] - np.array([0, 0])) < min_dist: continue
        if np.linalg.norm(test_pos[:2] - np.array([15, 15])) < min_dist: continue
        if all(np.linalg.norm(test_pos - p) > min_dist for p in positions):
            positions.append(test_pos)
    return positions

# -------------------- KALMAN FILTER --------------------
class DroneEstimator:
    def __init__(self, dt=1/240.):
        self.dt = dt
        self.x = np.zeros((6, 1))

        self.A = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ])

        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ])

        self.P = np.eye(6)
        self.Q = np.eye(6) * 1e-3
        self.R = np.eye(3) * SENSOR_STD**2
        self.I = np.eye(6)

    def update(self, z):
        z = np.asarray(z).reshape(3, 1)

        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q

        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (self.I - K @ self.H) @ self.P

        return self.x.flatten()

# -------------------- MAIN MISSION --------------------


def run_mission():
    p.resetSimulation()
    p.loadURDF("plane.urdf")

    OBSTACLE_HEIGHT = 2.0
    MAX_DRONE_Z = OBSTACLE_HEIGHT / 2.0

    start_pos = np.array([0, 0, MAX_DRONE_Z])
    end_pos = np.array([15, 15, MAX_DRONE_Z])

    create_body(start_pos - [0, 0, MAX_DRONE_Z - 0.1], [0.6, 0.6, 0.01], [0, 1, 0, 1])
    create_body(end_pos - [0, 0, MAX_DRONE_Z - 0.1], [0.8, 0.8, 0.01], [0, 0, 1, 1])

    obs_info = []
    for pos in get_random_positions(5):
        oid = create_body([pos[0], pos[1], OBSTACLE_HEIGHT],
                          [0.8, 0.8, OBSTACLE_HEIGHT],
                          [1, 0, 0, 1])
        obs_info.append({'id': oid})

    drone_id = create_body(start_pos, [0.3, 0.3, 0.1],
                           [0.2, 0.9, 0.2, 1], mass=1.0)
    
    orn = p.getBasePositionAndOrientation(drone_id)[1]
    euler = np.array(p.getEulerFromQuaternion(orn))
    ang_vel = np.array(p.getBaseVelocity(drone_id)[1])
    
    Kp = np.array([30, 30, 0])   # roll, pitch, yaw
    Kd = np.array([5, 5, 0])
    
    torque = -Kp * euler - Kd * ang_vel
    
    p.applyExternalTorque(drone_id, -1, torque, p.WORLD_FRAME)

    kf = DroneEstimator()
    prev_pos = start_pos.copy()

    print("Mission started (STABLE VERSION)...")

    for _ in range(2000):
        raw_pos, _ = p.getBasePositionAndOrientation(drone_id)
        noisy_pos = raw_pos + np.random.normal(0, SENSOR_STD, 3)

        state = kf.update(noisy_pos)
        pos, vel = state[:3], state[3:]

        # -------------------- CAMERA GATING --------------------
        view = p.computeViewMatrixFromYawPitchRoll(pos, 6, 0, -30, 0, 2)
        proj = p.computeProjectionMatrixFOV(60, 1, 0.1, 20)
        _, _, _, _, mask = p.getCameraImage(64, 64, view, proj,
                                            renderer=p.ER_TINY_RENDERER)

        f_rep = np.zeros(3)
        obstacle_normals = []

        for obs in obs_info:
            if np.sum(np.array(mask) == obs['id']) > 20:
                o_p, _ = p.getBasePositionAndOrientation(obs['id'])

                rel = pos - np.array(o_p)
                rel[2] = 0
                d = max(np.linalg.norm(rel), D_MIN)

                if d < D_SAFE:
                    n = rel / d
                    mag = K_REP * (1/d - 1/D_SAFE)
                    mag = np.clip(mag, 0, F_REP_MAX)
                    f_rep += mag * n
                    obstacle_normals.append(n)

        # -------------------- ATTRACTION (PROJECTED) --------------------
        dynamic_goal = np.array([end_pos[0], end_pos[1], MAX_DRONE_Z])
        f_att = K_ATT * (dynamic_goal - pos)

        for n in obstacle_normals:
            f_att -= np.dot(f_att, n) * n  # remove collision direction

        # -------------------- ADAPTIVE DAMPING --------------------
        damping = D_BASE + D_OBS * np.linalg.norm(f_rep)

        force = f_att + f_rep - damping * vel + np.array([0, 0, 9.81])

        p.applyExternalForce(drone_id, -1, force, raw_pos, p.WORLD_FRAME)

        p.addUserDebugLine(prev_pos, raw_pos, [0, 1, 0], 3, 0.1)
        prev_pos = raw_pos
        p.resetDebugVisualizerCamera(15, 45, -35, raw_pos)

        if np.linalg.norm(raw_pos[:2] - end_pos[:2]) < 0.7:
            print("Mission successful (stable).")
            break

        p.stepSimulation()
        time.sleep(1 / 240.)

    p.disconnect()

run_mission()
