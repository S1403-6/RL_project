"""
feature_extractor.py
====================
Extracts a 32-dimensional stacked perception observation.

Single frame = 16 features:
  Goal:
    [distance, angle, visibility] -> 3

  Obstacles:
    [5 proximity values, 5 sector angles, density] -> 11

  Optical flow:
    [vx, vy] -> 2

Two consecutive frames:
    16 x 2 = 32 features

The policy receives only visual/perceptual information.
No target position, robot position, or robot yaw is used.
"""

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

        self.single_feature_dim = 16
        self.stack_frames = stack_frames

        self.feature_history = []

    def reset(self):

        self.prev_gray = None
        self.feature_history = []

    def extract_single_frame(self, rgba_frame):

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

        f_goal = self._extract_goal_features(rgb)

        f_obs = self._extract_obstacle_features(rgb)

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

    def extract_observation(self, rgba_frame):

        curr_feat = self.extract_single_frame(rgba_frame)

        if len(self.feature_history) == 0:

            for _ in range(self.stack_frames):
                self.feature_history.append(
                    curr_feat.copy()
                )

        else:

            self.feature_history.append(
                curr_feat.copy()
            )

            if len(self.feature_history) > self.stack_frames:
                self.feature_history.pop(0)

        observation = np.concatenate(
            self.feature_history,
            axis=0
        )

        if observation.shape[0] != 32:
            raise RuntimeError(
                f"Expected 32-D observation, "
                f"got {observation.shape}"
            )

        return observation.astype(np.float32)

    # ============================================================
    # GOAL DETECTION
    # ============================================================

    def _extract_goal_features(self, rgb):

        hsv = cv2.cvtColor(
            rgb,
            cv2.COLOR_RGB2HSV
        )

        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]

        green_mask = (
            (h >= 35)
            & (h <= 85)
            & (s >= 100)
            & (v >= 80)
        ).astype(np.uint8) * 255

        kernel = np.ones(
            (5, 5),
            np.uint8
        )

        green_mask = cv2.morphologyEx(
            green_mask,
            cv2.MORPH_OPEN,
            kernel
        )

        green_mask = cv2.morphologyEx(
            green_mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        num_labels, labels, stats, centroids = (
            cv2.connectedComponentsWithStats(
                green_mask,
                connectivity=8
            )
        )

        best_area = 0.0
        best_centroid = None
        best_bbox = None

        for label in range(1, num_labels):

            x = stats[label, cv2.CC_STAT_LEFT]
            y = stats[label, cv2.CC_STAT_TOP]
            w = stats[label, cv2.CC_STAT_WIDTH]
            h_box = stats[label, cv2.CC_STAT_HEIGHT]
            area = stats[label, cv2.CC_STAT_AREA]

            if area < 20:
                continue

            if w < 3 or h_box < 3:
                continue

            aspect_ratio = w / float(h_box)

            if aspect_ratio < 0.25 or aspect_ratio > 4.0:
                continue

            fill_ratio = area / float(w * h_box + 1e-6)

            if fill_ratio < 0.15:
                continue

            if area > best_area:

                best_area = float(area)

                best_centroid = centroids[label].copy()

                best_bbox = (
                    x,
                    y,
                    w,
                    h_box
                )

        # --------------------------------------------------------
        # Goal not visible
        # --------------------------------------------------------

        if best_centroid is None:

            return np.array(
                [
                    0.0,
                    0.0,
                    0.0
                ],
                dtype=np.float32
            )

        goal_cx = float(best_centroid[0])

        # --------------------------------------------------------
        # Angular position of visually detected goal
        # --------------------------------------------------------

        angle_error = np.arctan2(
            goal_cx - self.cx,
            self.fl
        )

        angle_error = np.clip(
            angle_error,
            -np.pi / 2.0,
            np.pi / 2.0
        )

        # --------------------------------------------------------
        # Distance estimate from apparent size
        # --------------------------------------------------------

        x, y, w, h_box = best_bbox

        apparent_diameter = max(
            1.0,
            0.5 * (w + h_box)
        )

        # Calibrated heuristic for the current 0.4 m goal.
        estimated_distance = (
            80.0 / apparent_diameter
        )

        estimated_distance = np.clip(
            estimated_distance,
            0.0,
            20.0
        )

        distance_normalized = (
            estimated_distance / 20.0
        )

        return np.array(
            [
                distance_normalized,
                angle_error,
                1.0
            ],
            dtype=np.float32
        )

    # ============================================================
    # OBSTACLE DETECTION
    # ============================================================

    def _extract_obstacle_features(self, rgb):

        hsv = cv2.cvtColor(
            rgb,
            cv2.COLOR_RGB2HSV
        )

        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]

        # OpenCV hue range:
        # red is near 0 and near 180.
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

        # Remove isolated pixels and small noise.
        kernel = np.ones(
            (3, 3),
            np.uint8
        )

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

        # --------------------------------------------------------
        # Connected components
        # --------------------------------------------------------

        num_labels, labels, stats, centroids = (
            cv2.connectedComponentsWithStats(
                red_mask,
                connectivity=8
            )
        )

        obstacle_mask = np.zeros_like(
            red_mask
        )

        for label in range(1, num_labels):

            x = stats[label, cv2.CC_STAT_LEFT]
            y = stats[label, cv2.CC_STAT_TOP]
            w = stats[label, cv2.CC_STAT_WIDTH]
            h_box = stats[label, cv2.CC_STAT_HEIGHT]
            area = stats[label, cv2.CC_STAT_AREA]

            # Ignore tiny red artifacts.
            if area < 25:
                continue

            if w < 3 or h_box < 3:
                continue

            # Extremely thin regions are usually noise.
            if w / float(h_box + 1e-6) > 8.0:
                continue

            if h_box / float(w + 1e-6) > 8.0:
                continue

            # Keep only sufficiently compact components.
            fill_ratio = area / float(
                w * h_box + 1e-6
            )

            if fill_ratio < 0.10:
                continue

            obstacle_mask[
                labels == label
            ] = 255

        # --------------------------------------------------------
        # Sector-based obstacle representation
        # --------------------------------------------------------

        n_cols = 5

        col_width = self.W / float(n_cols)

        obs_proximity = []

        obs_angles = []

        for c in range(n_cols):

            col_start = int(
                round(c * col_width)
            )

            col_end = int(
                round((c + 1) * col_width)
            )

            col_start = max(
                0,
                min(self.W, col_start)
            )

            col_end = max(
                0,
                min(self.W, col_end)
            )

            col_mask = obstacle_mask[
                :,
                col_start:col_end
            ]

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

            obs_proximity.append(
                float(proximity)
            )

            sector_center = (
                (col_start + col_end) / 2.0
            )

            angle = np.arctan2(
                sector_center - self.cx,
                self.fl
            )

            obs_angles.append(
                float(angle)
            )

        # --------------------------------------------------------
        # Overall obstacle density
        # --------------------------------------------------------

        density = (
            np.count_nonzero(obstacle_mask)
            / float(obstacle_mask.size + 1e-6)
        )

        density = np.clip(
            density * 4.0,
            0.0,
            1.0
        )

        return np.array(
            obs_proximity
            + obs_angles
            + [density],
            dtype=np.float32
        )

    # ============================================================
    # OPTICAL FLOW
    # ============================================================

    def _extract_optical_flow(self, gray):

        if self.prev_gray is None:

            self.prev_gray = gray.copy()

            return np.zeros(
                2,
                dtype=np.float32
            )

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

            return np.zeros(
                2,
                dtype=np.float32
            )

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

        next_pts, status, _ = (
            cv2.calcOpticalFlowPyrLK(
                self.prev_gray,
                gray,
                prev_pts,
                None,
                **lk_params
            )
        )

        if next_pts is None or status is None:

            self.prev_gray = gray.copy()

            return np.zeros(
                2,
                dtype=np.float32
            )

        status = status.reshape(-1)

        good_prev = prev_pts.reshape(-1, 2)[
            status == 1
        ]

        good_next = next_pts.reshape(-1, 2)[
            status == 1
        ]

        if len(good_prev) < 3:

            self.prev_gray = gray.copy()

            return np.zeros(
                2,
                dtype=np.float32
            )

        flow = (
            good_next
            - good_prev
        )

        mean_flow = np.mean(
            flow,
            axis=0
        )

        mean_flow = np.clip(
            mean_flow / 20.0,
            -1.0,
            1.0
        )

        self.prev_gray = gray.copy()

        return mean_flow.astype(
            np.float32
        )