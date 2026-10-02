
import numpy as np
import cv2


class CNNFeatureExtractor:
    def __init__(
        self,
        img_width=320,
        img_height=240,
        fov_deg=90,
        stack_frames=2
    ):
        self.W = img_width
        self.H = img_height
        self.fov = fov_deg
        self.cx = img_width / 2.0
        self.cy = img_height / 2.0
        self.fl = (
            (img_width / 2.0)
            / np.tan(np.deg2rad(fov_deg / 2.0))
        )

        self.prev_gray = None
        self.single_feature_dim = 14
        self.stack_frames = stack_frames
        self.feature_history = []

    def reset(self):
        self.prev_gray = None
        self.feature_history = []

    @staticmethod
    def _relative_goal(robot_pos, robot_yaw, goal_pos):
        dx = float(goal_pos[0] - robot_pos[0])
        dy = float(goal_pos[1] - robot_pos[1])

        c = np.cos(robot_yaw)
        s = np.sin(robot_yaw)

        rel_x = c * dx + s * dy
        rel_y = -s * dx + c * dy
        distance = np.hypot(rel_x, rel_y)
        angle = np.arctan2(rel_y, rel_x)

        return np.array(
            [
                np.clip(rel_x / 24.0, -1.5, 1.5),
                np.clip(rel_y / 24.0, -1.0, 1.0),
                np.clip(distance / 24.0, 0.0, 2.0),
                np.clip(angle / np.pi, -1.0, 1.0),
            ],
            dtype=np.float32
        )

    @staticmethod
    def _nearest_obstacle_info(
        robot_pos,
        robot_yaw,
        obstacle_positions
    ):
        if obstacle_positions is None or len(obstacle_positions) == 0:
            return 1.0, 0.0

        robot_xy = np.asarray(robot_pos[:2], dtype=np.float32)

        best_distance = np.inf
        best_bearing = 0.0

        c = np.cos(robot_yaw)
        s = np.sin(robot_yaw)

        for obstacle in obstacle_positions:
            obs_xy = np.asarray(obstacle[:2], dtype=np.float32)
            dx = float(obs_xy[0] - robot_xy[0])
            dy = float(obs_xy[1] - robot_xy[1])

            distance = np.hypot(dx, dy)
            if distance < best_distance:
                best_distance = distance
                rel_x = c * dx + s * dy
                rel_y = -s * dx + c * dy
                best_bearing = np.arctan2(rel_y, rel_x)

        return (
            float(np.clip(best_distance / 20.0, 0.0, 1.5)),
            float(np.clip(best_bearing / np.pi, -1.0, 1.0))
        )

    def extract_single_frame(
        self,
        rgba_frame,
        robot_pos,
        robot_yaw,
        goal_pos,
        obstacle_positions
    ):
        if rgba_frame is None:
            raise ValueError("rgba_frame is None")

        if rgba_frame.ndim != 3 or rgba_frame.shape[2] < 3:
            raise ValueError(
                f"Expected image shape (H,W,3/4), got {rgba_frame.shape}"
            )

        rgb = rgba_frame[:, :, :3].copy()

        gray = cv2.cvtColor(
            rgb,
            cv2.COLOR_RGB2GRAY
        )

        f_goal = self._relative_goal(
            robot_pos,
            robot_yaw,
            goal_pos
        )

        f_obs = self._extract_obstacle_features(
            rgb,
            robot_pos,
            robot_yaw,
            obstacle_positions
        )

        f_flow = self._extract_optical_flow(gray)

        features = np.concatenate(
            [
                f_goal,
                f_obs,
                f_flow
            ]
        )

        if features.shape[0] != self.single_feature_dim:
            raise RuntimeError(
                f"Expected {self.single_feature_dim} features, "
                f"got {features.shape[0]}"
            )

        return features.astype(np.float32)

    def extract_observation(
        self,
        rgba_frame,
        robot_pos,
        robot_yaw,
        goal_pos,
        obstacle_positions
    ):
        curr_feat = self.extract_single_frame(
            rgba_frame,
            robot_pos,
            robot_yaw,
            goal_pos,
            obstacle_positions
        )

        if len(self.feature_history) == 0:
            for _ in range(self.stack_frames):
                self.feature_history.append(curr_feat.copy())
        else:
            self.feature_history.append(curr_feat.copy())
            if len(self.feature_history) > self.stack_frames:
                self.feature_history.pop(0)

        observation = np.concatenate(
            self.feature_history,
            axis=0
        )

        if observation.shape[0] != 28:
            raise RuntimeError(
                f"Expected 28-D observation, got {observation.shape}"
            )

        return observation.astype(np.float32)

    def _extract_obstacle_features(
        self,
        rgb,
        robot_pos,
        robot_yaw,
        obstacle_positions
    ):
        hsv = cv2.cvtColor(
            rgb,
            cv2.COLOR_RGB2HSV
        )

        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]

        red_mask_1 = (
            (h >= 0)
            & (h <= 10)
            & (s >= 120)
            & (v >= 80)
        )

        red_mask_2 = (
            (h >= 170)
            & (h <= 179)
            & (s >= 120)
            & (v >= 80)
        )

        red_mask = (
            red_mask_1 | red_mask_2
        ).astype(np.uint8) * 255

        kernel = np.ones((3, 3), np.uint8)

        red_mask = cv2.morphologyEx(
            red_mask,
            cv2.MORPH_OPEN,
            kernel
        )

        red_mask = cv2.morphologyEx(
            red_mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        num_labels, labels, stats, _ = (
            cv2.connectedComponentsWithStats(
                red_mask,
                connectivity=8
            )
        )

        obstacle_mask = np.zeros_like(red_mask)

        for label in range(1, num_labels):
            x = stats[label, cv2.CC_STAT_LEFT]
            y = stats[label, cv2.CC_STAT_TOP]
            w = stats[label, cv2.CC_STAT_WIDTH]
            h_box = stats[label, cv2.CC_STAT_HEIGHT]
            area = stats[label, cv2.CC_STAT_AREA]

            if area < 25 or w < 3 or h_box < 3:
                continue

            if w / float(h_box + 1e-6) > 8.0:
                continue

            if h_box / float(w + 1e-6) > 8.0:
                continue

            fill_ratio = area / float(w * h_box + 1e-6)
            if fill_ratio < 0.10:
                continue

            obstacle_mask[labels == label] = 255

        n_cols = 5
        col_width = self.W / float(n_cols)
        obs_proximity = []

        for c in range(n_cols):
            col_start = int(round(c * col_width))
            col_end = int(round((c + 1) * col_width))

            col_start = max(0, min(self.W, col_start))
            col_end = max(0, min(self.W, col_end))

            col_mask = obstacle_mask[:, col_start:col_end]

            if col_mask.size == 0:
                proximity = 0.0
            else:
                fraction = (
                    np.count_nonzero(col_mask)
                    / float(col_mask.size)
                )
                proximity = np.clip(
                    fraction * 6.0,
                    0.0,
                    1.0
                )

            obs_proximity.append(float(proximity))

        density = (
            np.count_nonzero(obstacle_mask)
            / float(obstacle_mask.size + 1e-6)
        )
        density = np.clip(density * 4.0, 0.0, 1.0)

        nearest_distance, nearest_bearing = (
            self._nearest_obstacle_info(
                robot_pos,
                robot_yaw,
                obstacle_positions
            )
        )

        return np.array(
            obs_proximity
            + [
                nearest_distance,
                nearest_bearing,
                density
            ],
            dtype=np.float32
        )

    def _extract_optical_flow(self, gray):
        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            return np.zeros(2, dtype=np.float32)

        feature_params = {
            "maxCorners": 30,
            "qualityLevel": 0.3,
            "minDistance": 7,
            "blockSize": 7
        }

        prev_pts = cv2.goodFeaturesToTrack(
            self.prev_gray,
            mask=None,
            **feature_params
        )

        if prev_pts is None or len(prev_pts) < 3:
            self.prev_gray = gray.copy()
            return np.zeros(2, dtype=np.float32)

        lk_params = {
            "winSize": (15, 15),
            "maxLevel": 2,
            "criteria": (
                cv2.TERM_CRITERIA_EPS
                | cv2.TERM_CRITERIA_COUNT,
                10,
                0.03
            )
        }

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            gray,
            prev_pts,
            None,
            **lk_params
        )

        if next_pts is None or status is None:
            self.prev_gray = gray.copy()
            return np.zeros(2, dtype=np.float32)

        status = status.reshape(-1)

        good_prev = prev_pts.reshape(-1, 2)[status == 1]
        good_next = next_pts.reshape(-1, 2)[status == 1]

        if len(good_prev) < 3:
            self.prev_gray = gray.copy()
            return np.zeros(2, dtype=np.float32)

        flow = good_next - good_prev
        mean_flow = np.mean(flow, axis=0)
        mean_flow = np.clip(mean_flow / 20.0, -1.0, 1.0)

        self.prev_gray = gray.copy()
        return mean_flow.astype(np.float32)
