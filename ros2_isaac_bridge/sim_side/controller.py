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


class HLInterfaceController(Node):
    def __init__(self):
        super().__init__("controller")

        # ---------------- ROS I/O ----------------
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
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

        self.sequence_of_objects = [0, 1, 2, 3, 4]
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
        return self.sequence_of_objects[self.sequence_index]

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

    def detect_object_orb(self) -> Optional[dict]:
        if self.latest_rgb is None:
            return None

        frame_gray = cv2.cvtColor(self.latest_rgb, cv2.COLOR_RGB2GRAY)
        kp_frame, des_frame = self.orb.detectAndCompute(frame_gray, None)

        if des_frame is None or len(kp_frame) < 20:
            return None

        best_detection = None
        best_score = -1.0

        frame_h, frame_w = frame_gray.shape[:2]

        for temp in self.templates:
            if temp["des"] is None:
                continue

            matches_knn = self.bf.knnMatch(temp["des"], des_frame, k=2)

            good_matches = []
            for pair in matches_knn:
                if len(pair) < 2:
                    continue
                m, n = pair
                if m.distance < 0.78 * n.distance:
                    good_matches.append(m)

            if len(good_matches) < 8:
                continue

            src_pts = np.float32([temp["kp"][m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if H is None or mask is None:
                continue

            inliers = int(mask.sum())
            if inliers < 6:
                continue

            w = temp["width"]
            h = temp["height"]

            template_corners = np.float32([
                [0, 0],
                [w - 1, 0],
                [w - 1, h - 1],
                [0, h - 1],
            ]).reshape(-1, 1, 2)

            projected_corners = cv2.perspectiveTransform(template_corners, H).reshape(-1, 2)

            xs = projected_corners[:, 0]
            ys = projected_corners[:, 1]

            x1 = int(np.clip(np.min(xs), 0, frame_w - 1))
            y1 = int(np.clip(np.min(ys), 0, frame_h - 1))
            x2 = int(np.clip(np.max(xs), 0, frame_w - 1))
            y2 = int(np.clip(np.max(ys), 0, frame_h - 1))

            bbox_w = x2 - x1
            bbox_h = y2 - y1

            if bbox_w < 20 or bbox_h < 20:
                continue

            if bbox_w > frame_w * 0.95 or bbox_h > frame_h * 0.95:
                continue

            score = float(inliers) / max(1, len(good_matches))

            if score > best_score:
                best_score = score
                best_detection = {
                    "id": temp["id"],
                    "name": temp["name"],
                    "bbox": (x1, y1, x2, y2),
                    "center_x": int((x1 + x2) / 2),
                    "center_y": int((y1 + y2) / 2),
                    "score": score,
                    "num_inliers": inliers,
                }

        return best_detection

    def get_depth_at_bbox_center(self, bbox: tuple, window: int = 5) -> Optional[float]:
        """
        Берет медианную глубину в маленьком окне вокруг центра bbox.
        """
        if self.latest_depth is None:
            return None

        x1, y1, x2, y2 = bbox
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        h, w = self.latest_depth.shape[:2]

        x_min = max(0, cx - window)
        x_max = min(w, cx + window + 1)
        y_min = max(0, cy - window)
        y_max = min(h, cy + window + 1)

        patch = self.latest_depth[y_min:y_max, x_min:x_max].copy()
        patch = np.nan_to_num(patch, nan=np.inf, posinf=np.inf, neginf=np.inf)

        valid = patch[np.isfinite(patch)]
        valid = valid[(valid > 0.1) & (valid < 10.0)]

        if len(valid) == 0:
            return None

        return float(np.median(valid))

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

    def detect_object_two_stage(self) -> Optional[dict]:
        candidate = self.detect_cube_face_candidate()
        if candidate is None:
            self.get_logger().info("two_stage: no face candidate")
            return None

        self.get_logger().info(f"two_stage: face candidate bbox={candidate['bbox']}")

        warped = self.warp_face_patch(candidate["quad"])
        if warped is None:
            self.get_logger().info("two_stage: warp failed")
            return None

        cls = self.classify_warped_face(warped)
        if cls is None:
            self.get_logger().info("two_stage: classification failed")
            return None

        x1, y1, x2, y2 = candidate["bbox"]

        return {
            "id": cls["id"],
            "name": cls["name"],
            "bbox": (x1, y1, x2, y2),
            "center_x": int((x1 + x2) / 2),
            "center_y": int((y1 + y2) / 2),
            "num_inliers": cls["num_inliers"],
            "score": cls["score"],
            "warped": warped,
        }

    def classify_warped_face(self, warped_rgb: np.ndarray) -> Optional[dict]:
        warped_gray = cv2.cvtColor(warped_rgb, cv2.COLOR_RGB2GRAY)
        kp_face, des_face = self.orb.detectAndCompute(warped_gray, None)

        if des_face is None or len(kp_face) < 12:
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
                if m.distance < 0.78 * n.distance:
                    good_matches.append(m)

            if len(good_matches) < 8:
                continue

            src_pts = np.float32([temp["kp"][m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_face[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
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

        # Ослабляем отрыв от второго места
        if second_best_inliers >= 0 and best_inliers - second_best_inliers < 2:
            return None

        best["score"] = float(best["num_inliers"]) / max(1, best["good_matches"])
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
        """
        Ищет в кадре прямоугольную/квадратную грань, похожую на грань куба.
        Возвращает:
        {
            "quad": np.ndarray shape (4,2),
            "bbox": (x1, y1, x2, y2)
        }
        или None
        """
        if self.latest_rgb is None:
            return None

        img = self.latest_rgb.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        edges = cv2.Canny(blur, 60, 160)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        h, w = gray.shape[:2]
        best = None
        best_area = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1200:
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)

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

            if bw < 30 or bh < 30:
                continue

            # Отношение сторон: грань куба примерно квадратная
            ratio = bw / max(1.0, bh)
            if ratio < 0.6 or ratio > 1.6:
                continue

            # Не берем почти весь кадр
            if bw > 0.9 * w or bh > 0.9 * h:
                continue

            # Берем самый большой правдоподобный четырехугольник
            if area > best_area:
                best_area = area
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

    def get_found_object_id(self) -> Optional[int]:
        if self.latest_rgb is None: return None
        gray = cv2.cvtColor(self.latest_rgb, cv2.COLOR_RGB2GRAY)
        _, des_frame = self.orb.detectAndCompute(gray, None)
        if des_frame is None: return None

        best_id, max_matches = None, 0
        for temp in self.templates:
            matches = self.bf.match(temp['des'], des_frame)
            good = [m for m in matches if m.distance < 40]
            if len(good) > 25 and len(good) > max_matches:
                max_matches = len(good)
                best_id = temp['id']
        return best_id

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
                self.send_command(-0.7, 0.0, 1.2)
                return
            else:
                self.is_recovering = False
                self.stuck_start_time = now
                self.get_logger().info("Восстановление завершено, возвращаюсь к EXPLORE")

        detection = self.detect_object_orb()
        obj_depth = None

        if detection is not None:
            obj_depth = self.get_depth_at_bbox_center(detection["bbox"])

            leader = self.update_majority_vote(detection)

            if leader is not None:
                self.get_logger().info(
                    f"OBJECT DETECTED | leader_id={leader['id']} | "
                    f"name={leader['name']} | votes={leader['count']} | depth={obj_depth}"
                )

                expected_id = self.get_expected_object_id()

                if obj_depth is not None and obj_depth <= self.approach_depth_threshold and leader["count"] >= 5:

                    if expected_id is not None and leader["id"] == expected_id:
                        self.log_detected_object(leader["id"], leader["name"], now)

                        self.get_logger().info(
                            f"SEQUENCE MATCHED | expected={expected_id} | found={leader['id']} | STOP 10 SEC"
                        )

                        self.sequence_index += 1
                        self.stop_until_time = now + 10.0
                        self.class_history.clear()

                    else:
                        self.log_detected_object(leader["id"], leader["name"], now)

        if now < self.stop_until_time:
            self.send_command(0.0, 0.0, 0.0)
            return

        if detection is not None and obj_depth is not None:
            frame_h, frame_w = self.latest_rgb.shape[:2]
            cx = detection["center_x"]
            err_x = (cx - frame_w / 2) / (frame_w / 2)

            if obj_depth > self.approach_depth_threshold:
                target_wz_obj = -0.8 * err_x
                target_vx_obj = 0.4 if abs(err_x) < 0.35 else 0.15

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
            vx, wz = -0.5, 1.8
        elif self.state == "EXPLORE":
            if dist_C < 1.2:
                vx = 0.15
                wz = 1.8 if dist_L > dist_R else -1.8
            elif dist_L < 0.7:
                vx = 1.5
                vy = -0.5
                wz = -0.6
            elif dist_R < 0.7:
                vx = 1.5
                vy = 0.5  # Валим боком
                wz = 0.6
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