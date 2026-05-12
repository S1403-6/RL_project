

import numpy as np
import cv2

class CNNFeatureExtractor:

    def __init__(self, img_width=320, img_height=240, fov_deg=90):
        self.W   = img_width
        self.H   = img_height
        self.fov = fov_deg
        self.cx  = img_width  / 2.0   
        self.cy  = img_height / 2.0   
        self.fl  = (img_width / 2.0) / np.tan(np.deg2rad(fov_deg / 2))

        self.prev_gray = None          
        self.feature_dim = 16

    def extract(self, rgba_frame):
        gray = cv2.cvtColor(rgba_frame, cv2.COLOR_RGBA2GRAY)
        rgb  = rgba_frame[:, :, :3]

        f_goal  = self._extract_goal_features(rgb)        # [dist, angle, visible]
        f_obs   = self._extract_obstacle_features(gray)   # [5 dists, 5 angles, density]
        f_flow  = self._extract_optical_flow(gray)        # [vx, vy]

        features = np.concatenate([f_goal, f_obs, f_flow])
        assert features.shape[0] == self.feature_dim, \
            f"Expected {self.feature_dim} features, got {features.shape[0]}"

        return features.astype(np.float32)

    def _extract_goal_features(self, rgb):
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        green_mask = ((g.astype(int) - r.astype(int) > 40) &
                      (g.astype(int) - b.astype(int) > 40) &
                      (g > 80))
        green_mask = green_mask.astype(np.uint8) * 255


        kernel     = np.ones((5, 5), np.uint8)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN,  kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

        moments = cv2.moments(green_mask)

        if moments["m00"] > 100:
            # centroid of green blob
            cx_blob = moments["m10"] / moments["m00"]
            cy_blob = moments["m01"] / moments["m00"]

            angle_error = np.arctan2(cx_blob - self.cx, self.fl)

            area          = moments["m00"] / 255.0
            dist_estimate = np.clip(1.0 / (np.sqrt(area) / self.W + 0.01), 0, 10.0)
            dist_estimate = dist_estimate / 10.0   # normalise to [0,1]

            goal_visible  = 1.0
        else:
            angle_error   = 0.0
            dist_estimate = 1.0   # assume far
            goal_visible  = 0.0

        return np.array([dist_estimate, angle_error, goal_visible], dtype=np.float32)



    def _extract_obstacle_features(self, gray):
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        # obstacle = darker than threshold (red boxes in dim scene)
        _, obs_mask = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY_INV)

        n_cols      = 5
        col_width   = self.W // n_cols
        obs_dists   = []
        obs_angles  = []

        for c in range(n_cols):
            col_start = c * col_width
            col_end   = col_start + col_width
            col_mask  = obs_mask[:, col_start:col_end]

            # fraction of column that is obstacle
            frac = np.sum(col_mask > 0) / (col_mask.size + 1e-6)

            # convert fraction → proximity (0=far, 1=very close)
            proximity = np.clip(frac * 5.0, 0.0, 1.0)
            obs_dists.append(proximity)

            # angle of this column's centre
            col_cx    = col_start + col_width / 2.0
            angle     = np.arctan2(col_cx - self.cx, self.fl)
            obs_angles.append(angle)

        # overall clutter density
        density = np.sum(obs_mask > 0) / (obs_mask.size + 1e-6)
        density = np.clip(density * 3.0, 0.0, 1.0)

        return np.array(obs_dists + obs_angles + [density], dtype=np.float32)


    def _extract_optical_flow(self, gray):
        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            return np.zeros(2, dtype=np.float32)

        # find good corners to track
        feature_params = dict(maxCorners=30, qualityLevel=0.3,
                              minDistance=7, blockSize=7)
        prev_pts = cv2.goodFeaturesToTrack(
            self.prev_gray, mask=None, **feature_params
        )

        if prev_pts is None or len(prev_pts) < 3:
            self.prev_gray = gray.copy()
            return np.zeros(2, dtype=np.float32)

        lk_params = dict(winSize=(15, 15), maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS |
                                   cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, prev_pts, None, **lk_params
        )

        good_prev = prev_pts[status == 1]
        good_next = next_pts[status == 1]

        if len(good_prev) == 0:
            self.prev_gray = gray.copy()
            return np.zeros(2, dtype=np.float32)

        flow      = good_next - good_prev       # shape (N, 2)
        mean_flow = flow.mean(axis=0)           # [fx, fy]

        # normalise: typical pixel flow ~5-20px per frame
        mean_flow = np.clip(mean_flow / 20.0, -1.0, 1.0)

        self.prev_gray = gray.copy()
        return mean_flow.astype(np.float32)



class SyntheticFrameGenerator:

    def __init__(self, img_width=320, img_height=240, fov_deg=90):
        self.W   = img_width
        self.H   = img_height
        self.fov = fov_deg
        self.fl  = (img_width / 2.0) / np.tan(np.deg2rad(fov_deg / 2))

    def render(self, robot_pos, robot_yaw, target_pos, obs_positions):
        frame = np.ones((self.H, self.W, 4), dtype=np.uint8) * 200
        frame[:, :, 3] = 255   # fully opaque

        # draw goal (green circle)
        goal_px = self._world_to_pixel(robot_pos, robot_yaw, target_pos)
        if goal_px is not None:
            cv2.circle(frame, goal_px, 20, (0, 220, 0, 255), -1)

        # draw obstacles (red rectangles)
        for obs in obs_positions:
            obs_px = self._world_to_pixel(robot_pos, robot_yaw, obs)
            if obs_px is not None:
                dist   = np.linalg.norm(np.array(robot_pos) - np.array(obs))
                height = int(np.clip(120 / (dist + 0.5), 10, self.H))
                width  = int(np.clip(60  / (dist + 0.5), 5,  80))
                x1 = max(0, obs_px[0] - width  // 2)
                y1 = max(0, obs_px[1] - height // 2)
                x2 = min(self.W, obs_px[0] + width  // 2)
                y2 = min(self.H, obs_px[1] + height // 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 30, 30, 255), -1)

        return frame

    def _world_to_pixel(self, robot_pos, robot_yaw, world_point):
        dx = world_point[0] - robot_pos[0]
        dy = world_point[1] - robot_pos[1]

        # rotate into camera frame
        cos_y =  np.cos(-robot_yaw)
        sin_y =  np.sin(-robot_yaw)
        cam_x =  cos_y * dx - sin_y * dy
        cam_z =  sin_y * dx + cos_y * dy   # forward axis

        if cam_z < 0.1:   # behind robot
            return None

        px = int(self.fl * (cam_x / cam_z) + self.W / 2)
        py = int(self.H / 2)   # objects appear at horizon height

        if 0 <= px < self.W:
            return (px, py)
        return None



if __name__ == "__main__":
    gen = SyntheticFrameGenerator()
    ext = CNNFeatureExtractor()

    frame = gen.render(
        robot_pos    = [0.0, 0.0],
        robot_yaw    = 0.0,
        target_pos   = [6.0, 0.0],
        obs_positions= [[1.8, 1.2], [1.8, -1.2], [2.8, 0.6]]
    )

    features = ext.extract(frame)

    print("Feature vector (16,):")
    labels = [
        "dist_to_goal", "goal_angle", "goal_visible",
        "obs_prox_col1", "obs_prox_col2", "obs_prox_col3",
        "obs_prox_col4", "obs_prox_col5",
        "obs_angle_col1","obs_angle_col2","obs_angle_col3",
        "obs_angle_col4","obs_angle_col5",
        "obstacle_density",
        "optical_flow_x", "optical_flow_y"
    ]
    for label, val in zip(labels, features):
        print(f"  {label:20s} = {val:.4f}")