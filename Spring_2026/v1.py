import pybullet as p
import pybullet_data
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt

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
        
        self.R = np.diag([meas_pos_std**2, meas_pos_std**2, meas_vel_std**2, meas_vel_std**2])

    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x

    def update(self, z):
        z = np.array(z).reshape(4, 1)
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.P.shape[0])
        self.P = np.dot((I - np.dot(K, self.H)), self.P)
        return self.x

class VisionSystem:
    def __init__(self, robot_id):
        self.robot_id = robot_id
        self.width = 320
        self.height = 240
        self.fov = 60
        # Pinhole Camera Math: f_px = (W/2) / tan(FOV/2)
        self.focal_length_px = (self.width / 2) / np.tan(np.deg2rad(self.fov / 2))
        
        self.prev_gray = None
        self.pixel_scaling_factor = 0.005 
    def capture_camera(self):
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        rot_matrix = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        
        cam_pos = np.array(pos) + np.array([0, 0, 0.3])
        target_pos = cam_pos + rot_matrix.dot(np.array([1, 0, 0]))
        
        view_matrix = p.computeViewMatrix(cam_pos, target_pos, [0, 0, 1])
        proj_matrix = p.computeProjectionMatrixFOV(self.fov, self.width/self.height, 0.02, 10)
        
        _, _, rgb, _, _ = p.getCameraImage(self.width, self.height, view_matrix, proj_matrix)
        rgb_array = np.array(rgb, dtype=np.uint8).reshape((self.height, self.width, 4))
        gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGBA2GRAY)
        
        return rgb_array, gray

    def estimate_velocity_from_flow(self, current_gray):
        if self.prev_gray is None:
            self.prev_gray = current_gray
            return 0.0, 0.0
            
        flow = cv2.calcOpticalFlowFarneback(self.prev_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        h, w = self.prev_gray.shape
        center_flow = flow[h//4:3*h//4, w//4:3*w//4]
        
        avg_dx = np.mean(center_flow[..., 0])
        avg_dy = np.mean(center_flow[..., 1])
        
        vx_est = avg_dx * self.pixel_scaling_factor
        vy_est = avg_dy * self.pixel_scaling_factor
        
        self.prev_gray = current_gray
        return vx_est, vy_est

    def estimate_obstacle_size(self, rgb_image, distance_to_obs):
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGBA2RGB)
        hsv = cv2.cvtColor(hsv, cv2.COLOR_RGB2HSV)
        
        mask1 = cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))
        mask = mask1 + mask2
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0
            
        c = max(contours, key=cv2.contourArea)
        x, y, w_px, h_px = cv2.boundingRect(c)
        
        real_width = (w_px * distance_to_obs) / self.focal_length_px
        
        return real_width


def lyapunov_controller(current_pos, current_vel, target_pos, obstacles):
    # Gains
    Kp = 2.5 
    Kd = 1.2 
    pos_error = np.array(target_pos[:2]) - np.array(current_pos[:2])
    dist_to_goal = np.linalg.norm(pos_error)
    
    if dist_to_goal < 0.1: return [0,0], dist_to_goal

    u_att = (pos_error * Kp) - (np.array(current_vel) * Kd)

    u_rep = np.array([0.0, 0.0])
    
    robot_radius = 0.3
    obs_radius = 0.35
    min_dist = robot_radius + obs_radius
    detection_range = 1.0
    
    for obs in obstacles:
        obs_vec = np.array(current_pos[:2]) - np.array(obs[:2])
        d_center = np.linalg.norm(obs_vec)
        d_surface = d_center - min_dist
        
        if d_surface < detection_range:
            if d_surface <= 0.01: d_surface = 0.01
            
            rep_dir = obs_vec / d_center
            
            t1 = np.array([-rep_dir[1], rep_dir[0]])
            t2 = np.array([rep_dir[1], -rep_dir[0]])
            
            tangent = t1 if np.dot(t1, pos_error) > np.dot(t2, pos_error) else t2
            
            K_rep = 6.0 
            mag = K_rep * (1.0/d_surface - 1.0/detection_range) / (d_surface**2)
            
            u_rep += (rep_dir * mag) + (tangent * mag * 3.0)

    u_total = u_att + u_rep
    
    speed = np.linalg.norm(u_total)
    max_speed = 1.2
    if speed > max_speed:
        u_total = (u_total / speed) * max_speed
        
    return u_total, dist_to_goal


def main():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.loadURDF("plane.urdf")
    

    start_pos = [0, 0, 0.1]
    robot_id = p.createMultiBody(baseMass=1, 
                                 baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.1], rgbaColor=[0, 0, 1, 1]),
                                 baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.1]),
                                 basePosition=start_pos)
    
    target_pos = [6, 0, 0]
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=0.2, rgbaColor=[0, 1, 0, 1]), basePosition=target_pos)


    obs_ids = []
    initial_obs_positions = [
        [2, 0, 0.5],
        [3.5, 1.5, 0.5],
        [3.5, -1.5, 0.5],
        [5, 0, 0.5]
    ]
    
    for pos in initial_obs_positions:
         oid = p.createMultiBody(baseMass=0, 
                                 baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.25, 0.25, 0.5], rgbaColor=[1, 0, 0, 1]),
                                 baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.25, 0.25, 0.5]),
                                 basePosition=pos)
         obs_ids.append(oid)


    kf = KalmanFilter(dt=1/240.0)
    vision = VisionSystem(robot_id)
    V_history = []
    history = {'gt_x': [], 'gt_y': [], 'est_x': [], 'est_y': []}
    
    print("Simulating... Obstacles are now MOVING.")

    for i in range(5000):

        current_obs_positions = []
        for idx, oid in enumerate(obs_ids):
            base_pos = initial_obs_positions[idx]
            

            amplitude = 1.2
            freq = 0.003
            offset = 0 if idx % 2 == 0 else np.pi 
            
            new_y = base_pos[1] + amplitude * np.sin(i * freq + offset)
            
            p.resetBasePositionAndOrientation(oid, [base_pos[0], new_y, base_pos[2]], [0,0,0,1])
            
            current_obs_positions.append([base_pos[0], new_y, base_pos[2]])

        p.stepSimulation()
        
        gt_pos, gt_orn = p.getBasePositionAndOrientation(robot_id)
        gt_vel, _ = p.getBaseVelocity(robot_id)
        
        meas_pos_x = gt_pos[0] + np.random.normal(0, 0.15)
        meas_pos_y = gt_pos[1] + np.random.normal(0, 0.15)
        
        vel_x_vis, vel_y_vis = 0, 0
        if i % 5 == 0:
            rgb, gray = vision.capture_camera()
            
            vel_x_vis = gt_vel[0] + np.random.normal(0, 0.1) 
            vel_y_vis = gt_vel[1] + np.random.normal(0, 0.1)
            

            dists = [np.linalg.norm(np.array(gt_pos[:2]) - np.array(op[:2])) for op in current_obs_positions]
            closest_dist = min(dists)
            
            est_width = vision.estimate_obstacle_size(rgb, closest_dist)
            if i % 100 == 0:
                print(f"Est Obstacle Width: {est_width:.3f}m")

        kf.predict()
        state_est = kf.update([meas_pos_x, meas_pos_y, vel_x_vis, vel_y_vis])
        
        est_pos = [state_est[0][0], state_est[1][0]]
        est_vel = [state_est[2][0], state_est[3][0]]
        
        pos_err = np.array(est_pos) - np.array(target_pos[:2])
        vel = np.array(est_vel)
        V = 0.5 * np.dot(pos_err, pos_err) + 0.5 * np.dot(vel, vel)
        V_history.append(V)

        cmd_vel, dist = lyapunov_controller(est_pos, est_vel, target_pos, current_obs_positions)
        
        p.resetBaseVelocity(robot_id, linearVelocity=[cmd_vel[0], cmd_vel[1], 0])
        
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
    
    for ob in initial_obs_positions:
        plt.gca().add_patch(plt.Rectangle((ob[0]-0.25, -1.5), 0.5, 3.0, color='r', alpha=0.1, label='Obs Range' if ob==initial_obs_positions[0] else ""))
        
    plt.legend()
    plt.title("Sensor Fusion Path Tracking with Dynamic Obstacles")
    plt.show()

    plt.figure(figsize=(8,4))
    plt.plot(V_history, linewidth=2)
    plt.xlabel("Time step")
    plt.ylabel("V(x, v)")
    plt.title("Energy Lyapunov Function Decrease")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()