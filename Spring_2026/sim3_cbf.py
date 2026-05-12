import pybullet as p
import pybullet_data
import numpy as np
import time
import random

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

SENSOR_STD = 0.05
DT = 1 / 240.

K_GOAL = 1.2          
K_TRACK = 25.0        
K_DAMP = 6.0          
D_SAFE = 1.5          
V_MAX = 3.0          
F_MAX = 60.0

def create_body(pos, size, color, mass=0):
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=size)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=size, rgbaColor=color)
    return p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=pos
    )

def get_random_positions(n, min_dist=5.0):
    positions = []
    while len(positions) < n:
        p0 = np.array([random.uniform(2, 13), random.uniform(2, 13), 0])
        if np.linalg.norm(p0[:2]) < min_dist: continue
        if np.linalg.norm(p0[:2] - np.array([15, 15])) < min_dist: continue
        if all(np.linalg.norm(p0 - p) > min_dist for p in positions):
            positions.append(p0)
    return positions


class DroneEstimator:
    def __init__(self, dt=DT):
        self.x = np.zeros((6, 1))
        self.A = np.array([
            [1,0,0,dt,0,0],
            [0,1,0,0,dt,0],
            [0,0,1,0,0,dt],
            [0,0,0,1,0,0],
            [0,0,0,0,1,0],
            [0,0,0,0,0,1],
        ])
        self.H = np.array([
            [1,0,0,0,0,0],
            [0,1,0,0,0,0],
            [0,0,1,0,0,0],
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


def run_mission():
    p.resetSimulation()
    p.loadURDF("plane.urdf")

    OBSTACLE_HEIGHT = 2.0
    Z = OBSTACLE_HEIGHT / 2.0

    start = np.array([0, 0, Z])
    goal  = np.array([15, 15, Z])

    create_body(start - [0,0,Z-0.1], [0.6,0.6,0.01], [0,1,0,1])
    create_body(goal  - [0,0,Z-0.1], [0.8,0.8,0.01], [0,0,1,1])

    obstacles = []
    for pos in get_random_positions(5):
        oid = create_body(
            [pos[0], pos[1], OBSTACLE_HEIGHT],
            [0.8, 0.8, OBSTACLE_HEIGHT],
            [1,0,0,1]
        )
        obstacles.append(oid)

    drone = create_body(start, [0.3,0.3,0.1],
                        [0.2,0.9,0.2,1], mass=1.0)

    # ---- attitude stabilization (unchanged) ----
    p.changeDynamics(drone, -1, angularDamping=1.0)

    kf = DroneEstimator()
    prev_pos = start.copy()

    print("Mission started (Velocity Obstacles)...")

    for _ in range(3000):
        raw_pos, orn = p.getBasePositionAndOrientation(drone)
        noisy_pos = raw_pos + np.random.normal(0, SENSOR_STD, 3)

        state = kf.update(noisy_pos)
        pos, vel = state[:3], state[3:]

        view = p.computeViewMatrixFromYawPitchRoll(pos, 6, 0, -30, 0, 2)
        proj = p.computeProjectionMatrixFOV(60, 1, 0.1, 20)
        _, _, _, _, mask = p.getCameraImage(
            64, 64, view, proj, renderer=p.ER_TINY_RENDERER
        )

        v_goal = K_GOAL * (goal - pos)
        v_goal[2] = 0.0

        speed = np.linalg.norm(v_goal)
        if speed > V_MAX:
            v_goal = v_goal / speed * V_MAX

        v_safe = v_goal.copy()
        for oid in obstacles:
            if np.sum(np.array(mask) == oid) > 20:
                o_p, _ = p.getBasePositionAndOrientation(oid)

                rel = pos - np.array(o_p)
                rel[2] = 0
                d = np.linalg.norm(rel)

                if d < 3.0:
                    n = rel / max(d, 1e-6)

                    v_toward = np.dot(v_safe, -n)

                    if v_toward > (d - D_SAFE):
                        v_safe -= v_toward * (-n)

        force = K_TRACK * (v_safe - vel) - K_DAMP * vel
        force[2] += 9.81
        force = np.clip(force, -F_MAX, F_MAX)

        p.applyExternalForce(drone, -1, force, raw_pos, p.WORLD_FRAME)

        p.addUserDebugLine(prev_pos, raw_pos, [0,1,0], 3, 0.1)
        prev_pos = raw_pos
        p.resetDebugVisualizerCamera(15, 45, -35, raw_pos)

        if np.linalg.norm(raw_pos[:2] - goal[:2]) < 1.2:
            print("Goal reached safely (VO).")
            break
        print("Distance to goal:", np.linalg.norm(raw_pos[:2] - goal[:2]))

        p.stepSimulation()
        time.sleep(DT)

    p.disconnect()

run_mission()
