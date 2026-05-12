# import pybullet as p
# import pybullet_data
# import time
# import numpy as np
# import matplotlib.pyplot as plt
# import cv2

# # ---------- EKF FUNCTION ----------
# def ekf_step(x_est, P, z, dt, Q, R):
#     x, y, th, v, w = x_est.flatten()

#     x_pred = np.array([
#         [x + v * np.cos(th) * dt],
#         [y + v * np.sin(th) * dt],
#         [th + w * dt],
#         [v],
#         [w]
#     ])

#     F = np.array([
#         [1, 0, -v*np.sin(th)*dt, np.cos(th)*dt, 0],
#         [0, 1,  v*np.cos(th)*dt, np.sin(th)*dt, 0],
#         [0, 0, 1,                0,           dt],
#         [0, 0, 0,                1,           0],
#         [0, 0, 0,                0,           1]
#     ])

#     P_pred = F @ P @ F.T + Q

#     H = np.array([
#         [1, 0, 0, 0, 0],
#         [0, 1, 0, 0, 0],
#         [0, 0, 0, 1, 0],
#         [0, 0, 0, 0, 1]
#     ])

#     y_k = z - H @ x_pred
#     S = H @ P_pred @ H.T + R
#     K = P_pred @ H.T @ np.linalg.inv(S)

#     x_new = x_pred + K @ y_k
#     P_new = (np.eye(5) - K @ H) @ P_pred

#     return x_new, P_new

# p.connect(p.GUI)
# p.setAdditionalSearchPath(pybullet_data.getDataPath())
# p.setGravity(0, 0, -9.8)

# plane = p.loadURDF("plane.urdf")
# robot = p.loadURDF("husky/husky.urdf", [0, 0, 0.1])

# # Add a visible obstacle
# obstacle = p.loadURDF("cube_small.urdf", [5, 0, 0.1])

# left_wheels  = [2, 3]
# right_wheels = [4, 5]

# for _ in range(100):
#     p.stepSimulation()
#     time.sleep(1/240)

# # ---------- EKF INIT ----------
# x_est = np.zeros((5, 1))
# P = np.eye(5) * 0.1
# Q = np.diag([0.01, 0.01, 0.01, 0.05, 0.05])
# R = np.diag([0.2, 0.2, 0.05, 0.01])
# dt = 1/60

# true_traj, gps_traj, ekf_traj = [], [], []

# # Camera parameters
# W, H = 320, 240
# fov = 60
# aspect = W / H
# near, far = 0.1, 10

# prev_cx = None

# start = time.time()

# while True:
#     for j in left_wheels + right_wheels:
#         p.setJointMotorControl2(robot, j, p.VELOCITY_CONTROL, targetVelocity=20, force=100)

#     p.stepSimulation()

#     pos, orn = p.getBasePositionAndOrientation(robot)
#     lin_vel, ang_vel = p.getBaseVelocity(robot)

#     encoder_v = lin_vel[0] + np.random.normal(0, 0.05)
#     imu_w = ang_vel[2] + np.random.normal(0, 0.01)
#     gps_x = pos[0] + np.random.normal(0, 0.2)
#     gps_y = pos[1] + np.random.normal(0, 0.2)

#     z = np.array([[gps_x], [gps_y], [encoder_v], [imu_w]])
#     x_est, P = ekf_step(x_est, P, z, dt, Q, R)

#     rot = p.getMatrixFromQuaternion(orn)
#     forward = [rot[0], rot[3], rot[6]]
#     cam_eye = pos
#     cam_target = [pos[0] + forward[0], pos[1] + forward[1], pos[2] + forward[2]]

#     view = p.computeViewMatrix(cam_eye, cam_target, [0,0,1])
#     proj = p.computeProjectionMatrixFOV(fov, aspect, near, far)
#     img = p.getCameraImage(W, H, view, proj)
#     rgb = np.reshape(img[2], (H, W, 4))[:, :, :3]

#     gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
#     _, th = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
#     contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     if contours:
#         c = max(contours, key=cv2.contourArea)
#         x, y, w, h = cv2.boundingRect(c)
#         cx = x + w/2

#         if prev_cx is not None:
#             dx = cx - prev_cx
#             pixel_v = dx / dt

#             dist = 2.0 
#             real_width = (w / W) * (2 * dist * np.tan(np.radians(fov/2)))
#             real_v = (pixel_v / W) * (2 * dist * np.tan(np.radians(fov/2)))

#             print("Obstacle width (m):", round(real_width,2),
#                   "Relative velocity (m/s):", round(real_v,2))

#         prev_cx = cx

#     true_traj.append([pos[0], pos[1]])
#     gps_traj.append([gps_x, gps_y])
#     ekf_traj.append([x_est[0,0], x_est[1,0]])

#     p.resetDebugVisualizerCamera(3, 50, -30, pos)
#     time.sleep(dt)

# # ---------- PLOT ----------
# true_traj = np.array(true_traj)
# gps_traj = np.array(gps_traj)
# ekf_traj = np.array(ekf_traj)

# plt.figure()
# plt.plot(true_traj[:,0], true_traj[:,1], label="Ground Truth")
# plt.plot(gps_traj[:,0], gps_traj[:,1], '.', alpha=0.4, label="Noisy GPS")
# plt.plot(ekf_traj[:,0], ekf_traj[:,1], label="EKF Estimate")
# plt.legend()
# plt.title("EKF State Estimation")
# plt.show()


