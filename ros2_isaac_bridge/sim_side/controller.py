#!/usr/bin/env python3
import math
import random
import numpy as np
import rclpy
import cv2
from typing import Dict, List, Optional, Tuple
from rclpy.node import Node
from collections import deque, Counter
from utils import OccupancyGrid, AStarPlanner
from geometry_msgs.msg import Twist, TwistStamped
from sensor_msgs.msg import Image, Imu, JointState
from std_msgs.msg import Int32, Int32MultiArray


class HLInterfaceController(Node):
    def __init__(self):
        super().__init__("controller")

        # ---------------- ROS I/O ----------------
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.detected_object_pub = self.create_publisher(Int32, "/competition/detected_object", 10)
        self.vel_sub = self.create_subscription(TwistStamped, "/aliengo/base_velocity", self._vel_callback, 10)
        self.joint_sub = self.create_subscription(JointState, "/aliengo/joint_states", self._joint_callback, 10)
        self.imu_sub = self.create_subscription(Imu, "/aliengo/imu", self._imu_callback, 10)
        self.rgb_sub = self.create_subscription(Image, "/aliengo/camera/color/image_raw", self._rgb_callback, 10)
        self.depth_sub = self.create_subscription(Image, "/aliengo/camera/depth/image_raw", self._depth_callback, 10)

        # --- Логика детектирования застревания ---
        self.last_valid_x = 0.0
        self.last_valid_y = 0.0
        self.stuck_start_time = self._now_sec()
        self.is_recovering = False
        self.recovery_end_time = 0.0

        self.class_history = deque(maxlen=15)  # история последних распознаваний
        self.last_majority_id = None
        self.last_majority_name = None

        self.detected_objects = {}  # уже подтвержденные объекты
        self.stop_until_time = 0.0  # если > now, стоим
        self.approach_depth_threshold = 0.9  # считаем, что подошли вплотную

        self.near_stop_duration = 2.0
        self.near_stop_depth_threshold = 0.5
        self.near_stopped_object_ids = set()

        self.sequence_of_objects = [
            (4, "laptop"),
            (3, "cup"),
            (1, "bottle"),
            (2, "chair"),
            (0, "backpack"),
        ]
        self.sequence_index = 0

        # ---------------- Cached state ----------------
        self.latest_base_velocity = {"vx": 0.0, "vy": 0.0, "wz": 0.0, "stamp_sec": None}
        self.latest_joint_state = {"names": [], "position": [], "velocity": [], "name_to_index": {}, "stamp_sec": None}
        self.latest_imu = {"wx": 0.0, "wy": 0.0, "wz": 0.0, "stamp_sec": None}
        self.latest_rgb: Optional[np.ndarray] = None
        self.latest_depth: Optional[np.ndarray] = None

        # ---------------- Навигация и Одометрия ----------------
        self.get_logger().info("Инициализация карты и планировщика...")
        self.nav_grid = OccupancyGrid(width_m=100.0, height_m=100.0, resolution=0.1, inflation_radius=0.35)
        self.planner = AStarPlanner(self.nav_grid)

        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.last_update_time = self._now_sec()

        self.state = "EXPLORE"
        self.current_path = []
        self.demo_enabled = True
        self.last_map_print_time = self._now_sec()
        self.log_period = 1.0
        self.last_log_time = 0.0

        self.orb = cv2.ORB_create(nfeatures=1200)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.face_warp_size = 220
        self.templates = []
        self._load_templates()

        self.create_timer(0.05, self._main_loop)
        self.get_logger().info("Controller started. Режим: Гибридная навигация + ORB Vision.")

    def get_expected_object_id(self) -> Optional[int]:
        if self.sequence_index >= len(self.sequence_of_objects):
            return None

        entry = self.sequence_of_objects[self.sequence_index]

        if isinstance(entry, (tuple, list)):
            return int(entry[0])

        return int(entry)

    def get_expected_object_name(self) -> Optional[str]:
        if self.sequence_index >= len(self.sequence_of_objects):
            return None

        entry = self.sequence_of_objects[self.sequence_index]

        if isinstance(entry, (tuple, list)) and len(entry) > 1:
            return str(entry[1])

        return None

    def publish_detected_object(self, object_id: int) -> None:
        msg = Int32()
        msg.data = int(object_id)
        self.detected_object_pub.publish(msg)

    def log_detected_object(self, object_id: int, object_name: str, t: float) -> None:
        if object_id in self.detected_objects:
            return

        self.publish_detected_object(object_id)

        self.detected_objects[object_id] = {
            "name": object_name,
            "t": round(t, 3),
            "x": round(self.robot_x, 4),
            "y": round(self.robot_y, 4),
            "yaw": round(self.robot_yaw, 4),
        }

        self.get_logger().info(
            f"OBJECT SAVED | id={object_id} | name={object_name} | "
            f"x={self.robot_x:.2f} y={self.robot_y:.2f} yaw={self.robot_yaw:.2f}"
        )

    def update_majority_vote(self, detection: dict) -> Optional[dict]:
        if detection is None:
            return None

        self.class_history.append((detection["id"], detection["name"]))

        if len(self.class_history) == 0:
            return None

        ids = [x[0] for x in self.class_history]
        names = {}
        for obj_id, obj_name in self.class_history:
            names[obj_id] = obj_name

        counts = Counter(ids)
        leader_id, leader_count = counts.most_common(1)[0]

        return {
            "id": leader_id,
            "name": names[leader_id],
            "count": leader_count,
        }

    def _load_templates(self):
        base_path = '/workspace/aliengo_competition/resources/assets/objects/'
        paths = [
            'backpack/backpack.jpg',
            'bottle/bottle.png',
            'chair/chair.png',
            'cup/cup.jpg',
            'laptop/laptop.png',
        ]

        self.templates = []

        for i, p in enumerate(paths):
            full_p = base_path + p
            img_bgr = cv2.imread(full_p)

            if img_bgr is None:
                self.get_logger().warn(f"Не найден файл объекта: {full_p}")
                continue

            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            img_gray = cv2.resize(img_gray, (self.face_warp_size, self.face_warp_size))

            kp, des = self.orb.detectAndCompute(img_gray, None)

            if des is None or len(kp) < 20:
                self.get_logger().warn(f"Слишком мало признаков у шаблона: {full_p}")
                continue

            h, w = img_gray.shape[:2]

            self.templates.append({
                "id": i,
                "name": p,
                "image_gray": img_gray,
                "kp": kp,
                "des": des,
                "width": w,
                "height": h,
            })

        self.get_logger().info(f"Загружено шаблонов: {len(self.templates)}")

    def order_quad_points(self, pts: np.ndarray) -> np.ndarray:
        """
        Упорядочивает 4 точки как:
        [top-left, top-right, bottom-right, bottom-left]
        """
        pts = pts.astype(np.float32)
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1).reshape(-1)

        top_left = pts[np.argmin(s)]
        bottom_right = pts[np.argmax(s)]
        top_right = pts[np.argmin(d)]
        bottom_left = pts[np.argmax(d)]

        return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)

    def classify_warped_face(self, warped_rgb: np.ndarray) -> Optional[dict]:
        warped_gray = cv2.cvtColor(warped_rgb, cv2.COLOR_RGB2GRAY)
        kp_face, des_face = self.orb.detectAndCompute(warped_gray, None)

        if des_face is None or len(kp_face) < 25:
            return None

        best = None
        best_inliers = -1
        best_score = -1.0
        second_best_score = -1.0

        for temp in self.templates:
            if temp["des"] is None or len(temp["kp"]) < 25:
                continue

            matches_knn = self.bf.knnMatch(temp["des"], des_face, k=2)

            good_matches = []
            for pair in matches_knn:
                if len(pair) < 2:
                    continue
                m, n = pair
                if m.distance < 0.72 * n.distance:
                    good_matches.append(m)

            if len(good_matches) < 12:
                continue

            src_pts = np.float32([temp["kp"][m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_face[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)
            if H is None or mask is None:
                continue

            inliers = int(mask.sum())
            if inliers < 10:
                continue

            inlier_ratio = inliers / max(1, len(good_matches))
            score = inliers + 8.0 * inlier_ratio

            if score > best_score:
                second_best_score = best_score
                best_score = score
                best_inliers = inliers
                best = {
                    "id": temp["id"],
                    "name": temp["name"],
                    "num_inliers": inliers,
                    "good_matches": len(good_matches),
                    "score": float(score),
                    "inlier_ratio": float(inlier_ratio),
                }
            elif score > second_best_score:
                second_best_score = score

        if best is None:
            return None

        if best["num_inliers"] < 10:
            return None

        if best["inlier_ratio"] < 0.45:
            return None

        if second_best_score >= 0 and best_score - second_best_score < 3.0:
            return None

        return best

    def warp_face_patch(self, quad: np.ndarray) -> Optional[np.ndarray]:
        """
        Берет найденный четырехугольник и выравнивает его в квадрат face_warp_size x face_warp_size
        """
        if self.latest_rgb is None:
            return None

        quad = self.order_quad_points(quad)

        dst = np.array([
            [0, 0],
            [self.face_warp_size - 1, 0],
            [self.face_warp_size - 1, self.face_warp_size - 1],
            [0, self.face_warp_size - 1],
        ], dtype=np.float32)

        H = cv2.getPerspectiveTransform(quad, dst)
        warped = cv2.warpPerspective(self.latest_rgb, H, (self.face_warp_size, self.face_warp_size))

        return warped

    def detect_cube_face_candidate(self) -> Optional[dict]:
        if self.latest_rgb is None:
            return None

        img = self.latest_rgb.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        edges = cv2.Canny(blur, 70, 180)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        h, w = gray.shape[:2]
        best = None
        best_score = -1.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 2500:
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.025 * peri, True)

            if len(approx) != 4:
                continue
            if not cv2.isContourConvex(approx):
                continue

            pts = approx.reshape(4, 2).astype(np.float32)

            xs = pts[:, 0]
            ys = pts[:, 1]
            x1, y1 = int(xs.min()), int(ys.min())
            x2, y2 = int(xs.max()), int(ys.max())

            bw = x2 - x1
            bh = y2 - y1
            if bw < 40 or bh < 40:
                continue

            if bw > 0.75 * w or bh > 0.75 * h:
                continue

            ratio = bw / max(1.0, bh)
            if ratio < 0.75 or ratio > 1.33:
                continue

            # Проверка, что стороны не слишком перекошены
            ordered = self.order_quad_points(pts)
            tl, tr, br, bl = ordered

            top = np.linalg.norm(tr - tl)
            right = np.linalg.norm(br - tr)
            bottom = np.linalg.norm(br - bl)
            left = np.linalg.norm(bl - tl)

            if min(top, right, bottom, left) < 25:
                continue

            side_ratio_1 = top / max(1.0, bottom)
            side_ratio_2 = left / max(1.0, right)

            if side_ratio_1 < 0.6 or side_ratio_1 > 1.7:
                continue
            if side_ratio_2 < 0.6 or side_ratio_2 > 1.7:
                continue

            rectangularity = area / max(1.0, bw * bh)
            if rectangularity < 0.65:
                continue

            score = area * rectangularity

            if score > best_score:
                best_score = score
                best = {
                    "quad": pts,
                    "bbox": (x1, y1, x2, y2),
                }

        return best

    # =====================================================================
    # High-level API
    # =====================================================================
    def send_command(self, vx: float, vy: float, wz: float) -> None:
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.angular.z = float(vx), float(vy), float(wz)
        self.cmd_pub.publish(msg)

    def detect_cube_face_candidates(self) -> List[dict]:
        if self.latest_rgb is None:
            return []

        image = self.latest_rgb
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape[:2]

        roi_y0 = int(h * 0.10)
        roi = gray[roi_y0:, :]

        blur = cv2.GaussianBlur(roi, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 140)

        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 250:
                continue

            hull = cv2.convexHull(cnt)
            rect = cv2.minAreaRect(hull)
            (_, _), (rw, rh), _ = rect

            if min(rw, rh) < 18:
                continue

            aspect = max(rw, rh) / max(1.0, min(rw, rh))
            if aspect > 2.4:
                continue

            box = cv2.boxPoints(rect).astype(np.float32)
            box[:, 1] += roi_y0

            xs = box[:, 0]
            ys = box[:, 1]

            x1 = int(np.clip(np.min(xs), 0, w - 1))
            y1 = int(np.clip(np.min(ys), 0, h - 1))
            x2 = int(np.clip(np.max(xs), 0, w - 1))
            y2 = int(np.clip(np.max(ys), 0, h - 1))

            bw = x2 - x1
            bh = y2 - y1
            if bw < 12 or bh < 12:
                continue

            rect_area = max(rw * rh, 1.0)
            extent = float(area) / rect_area
            if extent < 0.35:
                continue

            bbox = (x1, y1, x2, y2)
            depth = self.get_depth_at_rgb_bbox_center(bbox)

            if depth is not None:
                if not (0.2 <= depth <= 5.0):
                    continue

                width_m, height_m = self.estimate_bbox_size_m(bbox, depth)

                if width_m < 0.03 or height_m < 0.03:
                    continue
                if width_m > 1.2 or height_m > 1.2:
                    continue
            else:
                width_m, height_m = None, None

            lower_bonus = float((y1 + y2) * 0.5) / max(1.0, h)
            compact_bonus = 1.0 / aspect
            area_bonus = min(1.0, area / 4000.0)

            geom_score = 0.35 * compact_bonus + 0.30 * extent + 0.20 * lower_bonus + 0.15 * area_bonus

            candidates.append(
                {
                    "quad": box,
                    "bbox": bbox,
                    "depth": depth,
                    "geom_score": float(geom_score),
                    "extent": float(extent),
                    "aspect": float(aspect),
                    "width_m": width_m,
                    "height_m": height_m,
                }
            )

        candidates.sort(key=lambda c: c["geom_score"], reverse=True)
        return candidates[:8]

    def detect_cube_face_candidate(self) -> Optional[dict]:
        candidates = self.detect_cube_face_candidates()
        if not candidates:
            return None
        return candidates[0]

    def classify_warped_face(self, warped_rgb: np.ndarray) -> Optional[dict]:
        warped_gray = cv2.cvtColor(warped_rgb, cv2.COLOR_RGB2GRAY)
        kp_face, des_face = self.orb.detectAndCompute(warped_gray, None)

        if des_face is None or len(kp_face) < 14:
            return None

        best = None
        best_inliers = -1
        second_best_inliers = -1

        for temp in self.templates:
            matches_knn = self.bf.knnMatch(temp["des"], des_face, k=2)

            good_matches = []
            for pair in matches_knn:
                if len(pair) < 2:
                    continue
                m, n = pair
                if m.distance < 0.76 * n.distance:
                    good_matches.append(m)

            if len(good_matches) < 8:
                continue

            src_pts = np.float32([temp["kp"][m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_face[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)
            if H is None or mask is None:
                continue

            inliers = int(mask.sum())
            if inliers < 6:
                continue

            if inliers > best_inliers:
                second_best_inliers = best_inliers
                best_inliers = inliers
                best = {
                    "id": temp["id"],
                    "name": temp["name"],
                    "num_inliers": inliers,
                    "good_matches": len(good_matches),
                }
            elif inliers > second_best_inliers:
                second_best_inliers = inliers

        if best is None:
            return None

        if second_best_inliers >= 0 and best_inliers - second_best_inliers < 3:
            return None

        best["score"] = float(best["num_inliers"]) / max(1, best["good_matches"])
        if best["score"] < 0.38:
            return None

        return best

    def estimate_bbox_size_m(self, bbox: tuple, depth_m: float) -> tuple:
        if self.latest_rgb is None:
            return 999.0, 999.0

        x1, y1, x2, y2 = bbox
        rgb_h, rgb_w = self.latest_rgb.shape[:2]

        bw = max(1.0, float(x2 - x1))
        bh = max(1.0, float(y2 - y1))

        hfov = math.radians(70.0)
        vfov = 2.0 * math.atan((rgb_h / rgb_w) * math.tan(hfov / 2.0))

        width_m = 2.0 * depth_m * math.tan(hfov / 2.0) * (bw / rgb_w)
        height_m = 2.0 * depth_m * math.tan(vfov / 2.0) * (bh / rgb_h)

        return width_m, height_m

    def get_depth_at_rgb_bbox_center(self, bbox: tuple, window: int = 5) -> Optional[float]:
        if self.latest_rgb is None or self.latest_depth is None:
            return None

        x1, y1, x2, y2 = bbox

        rgb_h, rgb_w = self.latest_rgb.shape[:2]
        depth_h, depth_w = self.latest_depth.shape[:2]

        cx_rgb = int((x1 + x2) / 2)
        cy_rgb = int((y1 + y2) / 2)

        cx = int(cx_rgb * depth_w / max(1, rgb_w))
        cy = int(cy_rgb * depth_h / max(1, rgb_h))

        x_min = max(0, cx - window)
        x_max = min(depth_w, cx + window + 1)
        y_min = max(0, cy - window)
        y_max = min(depth_h, cy + window + 1)

        patch = self.latest_depth[y_min:y_max, x_min:x_max].astype(np.float32)
        valid = patch[np.isfinite(patch)]
        valid = valid[(valid > 0.15) & (valid < 8.0)]

        if len(valid) == 0:
            return None
        return float(np.median(valid))

    def detect_object(self) -> Optional[dict]:
        candidates = self.detect_cube_face_candidates()
        if not candidates:
            return None

        best_detection = None
        best_total_score = -1e9

        for cand in candidates:
            warped = self.warp_face_patch(cand["quad"])
            if warped is None:
                continue

            cls = self.classify_warped_face(warped)
            if cls is None:
                continue

            x1, y1, x2, y2 = cand["bbox"]

            total_score = (
                    0.70 * float(cls["score"])
                    + 0.20 * float(cand["geom_score"])
                    + 0.10 * min(1.0, float(cls["num_inliers"]) / 15.0)
            )

            det = {
                "id": cls["id"],
                "name": cls["name"],
                "bbox": (x1, y1, x2, y2),
                "center_x": int((x1 + x2) / 2),
                "center_y": int((y1 + y2) / 2),
                "depth": cand["depth"],
                "score": float(cls["score"]),
                "num_inliers": int(cls["num_inliers"]),
                "geom_score": float(cand["geom_score"]),
                "total_score": float(total_score),
            }

            if total_score > best_total_score:
                best_total_score = total_score
                best_detection = det

        return best_detection

    def get_laser_scan(self) -> Optional[np.ndarray]:
        if self.latest_depth is None: return None
        h, w = self.latest_depth.shape

        row_idx = int(h * 0.35)

        raw_row = np.nanmedian(self.latest_depth[row_idx - 2: row_idx + 3, :], axis=0)

        hfov = math.radians(86.0)
        angles_offset = np.linspace(-hfov / 2, hfov / 2, w)
        corrected_row = raw_row / np.cos(angles_offset)

        corrected_row = np.nan_to_num(corrected_row, posinf=5.0, neginf=0.0)

        corrected_row[corrected_row > 4.0] = 5.0
        corrected_row[corrected_row < 0.2] = 5.0
        return corrected_row

    # =====================================================================
    # Main Logic
    # =====================================================================
    def run_user_code(self) -> None:
        now = self._now_sec()
        dt = now - self.last_update_time
        self.last_update_time = now
        if dt > 0.5: dt = 0.05

        # 1. Одометрия
        actual_vx, actual_wz = self.latest_base_velocity["vx"], self.latest_base_velocity["wz"]
        self.robot_yaw += actual_wz * dt
        self.robot_yaw = (self.robot_yaw + math.pi) % (2 * math.pi) - math.pi
        self.robot_x += actual_vx * math.cos(self.robot_yaw) * dt
        self.robot_y += actual_vx * math.sin(self.robot_yaw) * dt

        dist_moved = math.sqrt((self.robot_x - self.last_valid_x) ** 2 + (self.robot_y - self.last_valid_y) ** 2)

        if dist_moved > 0.2:
            self.last_valid_x = self.robot_x
            self.last_valid_y = self.robot_y
            self.stuck_start_time = now

        if not self.is_recovering and (now - self.stuck_start_time > 10.0):
            self.get_logger().warn("РОБОТ ЗАСТРЯЛ! Запускаю разворот...")
            self.is_recovering = True
            self.recovery_end_time = now + 5.0

        if self.is_recovering:
            if now < self.recovery_end_time:
                self.send_command(-1.5, 0.0, 2)
                return
            else:
                self.is_recovering = False
                self.stuck_start_time = now
                self.get_logger().info("Восстановление завершено, возвращаюсь к EXPLORE")

        detection = self.detect_object()
        obj_depth = detection["depth"] if detection is not None else None

        if detection is not None:
            leader = self.update_majority_vote(detection)

            if leader is not None:
                self.get_logger().info(
                    f"OBJECT DETECTED | leader_id={leader['id']} | "
                    f"name={leader['name']} | votes={leader['count']} | depth={obj_depth}"
                )

                expected_id = self.get_expected_object_id()
                expected_name = self.get_expected_object_name()

                # Останавливаемся только на правильном объекте по порядку
                if (
                        expected_id is not None
                        and leader["id"] == expected_id
                        and obj_depth is not None
                        and obj_depth <= self.near_stop_depth_threshold
                        and leader["count"] >= 3
                        and leader["id"] not in self.near_stopped_object_ids
                ):
                    self.near_stopped_object_ids.add(leader["id"])
                    self.log_detected_object(leader["id"], leader["name"], now)

                    self.stop_until_time = max(self.stop_until_time, now + self.near_stop_duration)
                    self.sequence_index += 1
                    self.class_history.clear()

                    self.get_logger().info(
                        f"SEQUENCE MATCHED | expected_id={expected_id} | expected_name={expected_name} | "
                        f"found_id={leader['id']} | found_name={leader['name']} | "
                        f"STOP {self.near_stop_duration:.1f} SEC"
                    )

        if now < self.stop_until_time:
            self.send_command(0.0, 0.0, 0.0)
            return

        if detection is not None and obj_depth is not None:
            frame_h, frame_w = self.latest_rgb.shape[:2]
            cx = detection["center_x"]
            err_x = (cx - frame_w / 2) / (frame_w / 2)

            if obj_depth > self.approach_depth_threshold:
                target_wz_obj = -2 * err_x
                target_vx_obj = 1 if abs(err_x) < 0.3 else 0.4

                self.send_command(target_vx_obj, 0.0, target_wz_obj)
                return

        scan = self.get_laser_scan()
        dist_L, dist_C, dist_R = 5.0, 5.0, 5.0
        if scan is not None:
            width = len(scan)
            hfov = math.radians(86.0)
            rel_angles = np.linspace(-hfov / 2, hfov / 2, width)

            for i in range(0, width, 15):
                dist, ang = scan[i], self.robot_yaw + rel_angles[i]
                if dist < 0.3: continue

                clear_dist = dist if dist < 4.0 else 4.0

                for r in np.arange(0.2, clear_dist, 0.2):
                    self.nav_grid.mark_free(self.robot_x + r * math.cos(ang),
                                            self.robot_y + r * math.sin(ang))

                if dist < 4.0:
                    ox = self.robot_x + dist * math.cos(ang)
                    oy = self.robot_y + dist * math.sin(ang)
                    self.nav_grid.mark_obstacle(ox, oy)

            # Обновляем зоны для управления (на основе 4-метрового скана)
            # Чтобы робот не тормозил слишком рано, можно оставить порог логики 1.0м
            dist_L = float(np.min(scan[:width // 3]))
            dist_C = float(np.min(scan[width // 3: 2 * width // 3]))
            dist_R = float(np.min(scan[2 * width // 3:]))

        # 4. Логика управления (Explore)
        vx, vy, wz = 0.0, 0.0, 0.0

        if self.is_recovering:
            vx, wz = -1.5, 2
        elif self.state == "EXPLORE":
            if dist_C < 1.2:
                vx = 0.2
                wz = 2 if dist_L > dist_R else -2
            elif dist_L < 0.7:
                vx = 1.5
                vy = -0.75
                wz = -1
            elif dist_R < 0.7:
                vx = 1.5
                vy = 0.75  # Валим боком
                wz = 1
            else:
                vx = 1.5
                vy = 0.0
                wz = 0.0

        self.send_command(vx, vy, wz)

        if now - self.last_map_print_time > 2.0:
            # self.print_local_map(radius_cells=30)  # вывод карты в консоль (аля 2д слем)
            self.last_map_print_time = now

    def save_full_map(self, filename="final_map_result.txt"):
        self.get_logger().info(f"Сохранение карты в {filename}...")
        grid = self.nav_grid.grid
        explored = (grid != -1)
        if not np.any(explored): return
        rows, cols = np.where(explored)
        min_r, max_r = max(0, np.min(rows) - 2), min(grid.shape[0] - 1, np.max(rows) + 2)
        min_c, max_c = max(0, np.min(cols) - 2), min(grid.shape[1] - 1, np.max(cols) + 2)
        sub = grid[min_r:max_r + 1, min_c:max_c + 1]
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"Size: {sub.shape[1] * 0.1:.1f}x{sub.shape[0] * 0.1:.1f}m\n")
            for r in range(sub.shape[0]):
                chars = []
                for c in range(sub.shape[1]):
                    v = sub[r, c]
                    if v == -1:
                        chars.append(' ')
                    elif v == 1:
                        chars.append('█')
                    else:
                        chars.append('.')
                f.write("".join(chars) + "\n")

    def print_local_map(self, radius_cells=20):
        r_row, r_col = self.nav_grid.world_to_grid(self.robot_x, self.robot_y)

        # self.get_logger().info(f"Robot Index: [{r_row}, {r_col}]")

        res = "\n=== LOCAL MAP (Centered) ===\n"
        res += "+" + "-" * (radius_cells * 2 + 1) + "+\n"

        for i in range(r_row - radius_cells, r_row + radius_cells + 1):
            row_chars = ["|"]
            for j in range(r_col - radius_cells, r_col + radius_cells + 1):
                if i == r_row and j == r_col:
                    row_chars.append('R')
                    continue

                if 0 <= i < self.nav_grid.rows and 0 <= j < self.nav_grid.cols:
                    val = self.nav_grid.grid[i, j]
                    if val == -1:
                        row_chars.append(' ')
                    elif val == 1:
                        row_chars.append('█')
                    else:
                        row_chars.append('.')
                else:
                    row_chars.append('▒')

            row_chars.append("|")
            res += "".join(row_chars) + "\n"

        res += "+" + "-" * (radius_cells * 2 + 1) + "+\n"
        self.get_logger().info(res)

    # ---------------- CALLBACKS & UTILS ----------------
    def _vel_callback(self, msg):
        self.latest_base_velocity = {"vx": msg.twist.linear.x, "vy": msg.twist.linear.y, "wz": msg.twist.angular.z,
                                     "stamp_sec": self._msg_time_to_sec(msg.header.stamp)}

    def _imu_callback(self, msg):
        self.latest_imu = {"wx": msg.angular_velocity.x, "wy": msg.angular_velocity.y, "wz": msg.angular_velocity.z,
                           "stamp_sec": self._msg_time_to_sec(msg.header.stamp)}

    def _joint_callback(self, msg):
        self.latest_joint_state = {"names": list(msg.name), "position": list(msg.position),
                                   "name_to_index": {n: i for i, n in enumerate(msg.name)},
                                   "stamp_sec": self._msg_time_to_sec(msg.header.stamp)}

    def _rgb_callback(self, msg):
        self.latest_rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))

    def _depth_callback(self, msg):
        self.latest_depth = np.frombuffer(msg.data, dtype=np.float32).reshape((msg.height, msg.width))

    def _main_loop(self):
        if self.demo_enabled: self.run_user_code()

    def _now_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _msg_time_to_sec(self, stamp):
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def robot_state_ready(self):
        return self.latest_base_velocity["stamp_sec"] is not None


def main(args=None):
    rclpy.init(args=args)
    node = HLInterfaceController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_full_map("final_map_result.txt")
        node.send_command(0, 0, 0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()