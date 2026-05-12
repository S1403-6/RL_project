import pybullet as p
import pybullet_data
import numpy as np
import time
import random


p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

def create_body(pos, size, color, mass=0):
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=size)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=size, rgbaColor=color)
    return p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=col, 
                            baseVisualShapeIndex=vis, basePosition=pos)

def get_random_positions(n, min_dist=5.0):
    positions = []
    while len(positions) < n:
        test_pos = np.array([random.uniform(2, 13), random.uniform(2, 13), 0])
        if np.linalg.norm(test_pos[:2] - np.array([0,0])) < min_dist: continue
        if np.linalg.norm(test_pos[:2] - np.array([15,15])) < min_dist: continue
        if all(np.linalg.norm(test_pos - p) > min_dist for p in positions):
            positions.append(test_pos)
    return positions

class DroneEstimator:
    def __init__(self, dt=1/240.):
        self.dt = dt

        # State: [x, y, z, vx, vy, vz]
        self.x = np.zeros((6, 1))

        # State transition matrix (constant velocity model)
        self.A = np.array([
            [1, 0, 0, dt, 0,  0],
            [0, 1, 0, 0,  dt, 0],
            [0, 0, 1, 0,  0,  dt],
            [0, 0, 0, 1,  0,  0],
            [0, 0, 0, 0,  1,  0],
            [0, 0, 0, 0,  0,  1],
        ])

        # Measurement matrix (we measure only position)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ])

        # Covariances
        self.P = np.eye(6) * 1.0          # State uncertainty
        self.Q = np.eye(6) * 1e-3         # Process noise
        self.R = np.eye(3) * 5e-3         # Measurement noise

        self.I = np.eye(6)

    def update(self, z):
        z = np.asarray(z).reshape(3, 1)

        # --- Predict ---
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q

        # --- Update ---
        y = z - (self.H @ self.x)                          # Innovation
        S = self.H @ self.P @ self.H.T + self.R            # Innovation cov
        K = self.P @ self.H.T @ np.linalg.inv(S)           # Kalman gain

        self.x = self.x + K @ y
        self.P = (self.I - K @ self.H) @ self.P

        return self.x.flatten()


def run_mission():
    p.resetSimulation()
    p.loadURDF("plane.urdf")
    
    OBSTACLE_HEIGHT = 2.0                 # Fixed height for all obstacles
    MAX_DRONE_Z = OBSTACLE_HEIGHT / 2.0   # Drone cannot go above this

    start_pos = np.array([0, 0, MAX_DRONE_Z])
    end_pos   = np.array([15, 15, MAX_DRONE_Z])
    
    create_body(start_pos - [0, 0, MAX_DRONE_Z-0.1], [0.6, 0.6, 0.01], [0, 1, 0, 1])
    create_body(end_pos   - [0, 0, MAX_DRONE_Z-0.1], [0.8, 0.8, 0.01], [0, 0, 1, 1])

    # Obstacles: SAME HEIGHT, NO RANDOMNESS
    obs_positions = get_random_positions(5)
    obs_info = []
    for pos in obs_positions:
        h = OBSTACLE_HEIGHT
        oid = create_body([pos[0], pos[1], h], [0.8, 0.8, h], [1, 0, 0, 1])
        obs_info.append({'id': oid, 'height': h})

    drone_id = create_body(start_pos, [0.3, 0.3, 0.1], [0.2, 0.9, 0.2, 1], mass=1.0)
    kf = DroneEstimator()
    prev_pos = start_pos.copy()

    while True:
        raw_pos, _ = p.getBasePositionAndOrientation(drone_id)
        state = kf.update(np.array(raw_pos))
        pos, vel = state[:3], state[3:]

        # --- FOV SIZE ESTIMATION ---
        view = p.computeViewMatrixFromYawPitchRoll(pos, 6, 0, -30, 0, 2)
        proj = p.computeProjectionMatrixFOV(60, 1, 0.1, 20)
        _, _, _, _, mask = p.getCameraImage(64, 64, view, proj, renderer=p.ER_TINY_RENDERER)
        
        target_height = MAX_DRONE_Z
        f_rep = np.array([0, 0, 0], dtype=float)

        for obs in obs_info:
            pixel_count = np.sum(np.array(mask) == obs['id'])
            if pixel_count > 40:
                # Drone is NOT allowed to climb over obstacles anymore
                target_height = MAX_DRONE_Z
                
                # Horizontal Repulsion (Lyapunov)
                o_p, _ = p.getBasePositionAndOrientation(obs['id'])
                rel = pos - np.array(o_p)
                rel[2] = 0
                d = np.linalg.norm(rel)
                f_rep += 120.0 * (rel / (d**5))

        # --- LYAPUNOV CONTROL LAW ---
        dynamic_goal = np.array([end_pos[0], end_pos[1], target_height])
        
        f_att = 3.0 * (dynamic_goal - pos)
        force = f_att + f_rep - 2.0 * vel + np.array([0, 0, 9.81])

        p.applyExternalForce(drone_id, -1, force, raw_pos, p.WORLD_FRAME)
        
        p.addUserDebugLine(prev_pos, raw_pos, [0, 1, 0], 3, 0.1)
        prev_pos = raw_pos
        p.resetDebugVisualizerCamera(15, 45, -35, raw_pos)

        if np.linalg.norm(raw_pos[:2] - end_pos[:2]) < 0.7:
            print("Mission successful! Clearing obstacles.")
            time.sleep(1)
            break
        
        p.stepSimulation()
        time.sleep(1./240.)


run_mission()