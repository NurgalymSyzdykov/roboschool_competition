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
from sensor_msgs.msg import Image, Imu, JointState, CameraInfo
from std_msgs.msg import Int32, Int32MultiArray


# --- Публикатор CameraInfo ---
class DepthCameraInfoPublisher(Node):
    def __init__(self):
        super().__init__("depth_camera_info_pub")

        self.pub = self.create_publisher(
            CameraInfo,
            "/aliengo/camera/depth/camera_info",
            10,
        )

        self.width = 848
        self.height = 480
        self.fov_h_deg = 86.0

        fov_h = math.radians(self.fov_h_deg)
        fx = (self.width / 2.0) / math.tan(fov_h / 2.0)
        fy = fx
        cx = self.width / 2.0
        cy = self.height / 2.0

        self.msg = CameraInfo()
        self.msg.header.frame_id = "front_camera_depth"
        self.msg.width = self.width
        self.msg.height = self.height
        self.msg.distortion_model = "plumb_bob"
        self.msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]

        self.msg.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        self.msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        self.msg.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]

        self.timer = self.create_timer(0.1, self.timer_callback)
        self.get_logger().info("Depth camera_info publisher started.")

    def timer_callback(self):
        self.msg.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(self.msg)


# --- Основной контроллер ---
class HLInterfaceController(Node):
    def __init__(self):
        super().__init__("controller")

        # ---------------- ROS I/O ----------------
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.detected_object_pub = self.create_publisher(Int32, "/competition/detected_object", 10)

        # Подписка на параметры камеры
        self.cam_sub = self.create_subscription(CameraInfo, "/camera/camera_info",
                                                self._cam_info_callback, 10)

        self.vel_sub = self.create_subscription(TwistStamped, "/odom", self._vel_callback, 10)
        self.imu_sub = self.create_subscription(Imu, "/imu", self._imu_callback, 10)
        self.rgb_sub = self.create_subscription(Image, "/camera/color/image_raw", self._rgb_callback, 10)
        self.depth_sub = self.create_subscription(Image, "/camera/depth/image_raw", self._depth_callback, 10)

        # --- Динамические параметры камеры ---
        self.hfov_rad = math.radians(86.0)  # Значение по умолчанию
        self.vfov_rad = math.radians(50.0)  # Значение по умолчанию
        self.cam_intrinsics_ready = False

        # --- Логика детектирования застревания ---
        self.last_valid_x = 0.0
        self.last_valid_y = 0.0
        self.stuck_start_time = self._now_sec()
        self.is_recovering = False
        self.recovery_end_time = 0.0

        self.class_history = deque(maxlen=15)
        self.detected_objects = {}
        self.stop_until_time = 0.0
        self.approach_depth_threshold = 0.9

        self.near_stop_duration = 2.0
        self.near_stop_depth_threshold = 0.5
        self.near_stopped_object_ids = set()

        self.sequence_of_objects = [
            (4, "laptop"), (3, "cup"), (1, "bottle"), (2, "chair"), (0, "backpack"),
        ]
        self.sequence_index = 0

        # ---------------- Cached state ----------------
        self.latest_base_velocity = {"vx": 0.0, "vy": 0.0, "wz": 0.0, "stamp_sec": None}
        self.latest_joint_state = {"names": [], "position": [], "velocity": [], "name_to_index": {}, "stamp_sec": None}
        self.latest_imu = {"wx": 0.0, "wy": 0.0, "wz": 0.0, "stamp_sec": None}
        self.latest_rgb: Optional[np.ndarray] = None
        self.latest_depth: Optional[np.ndarray] = None

        # ---------------- Навигация ----------------
        self.get_logger().info("Инициализация карты и планировщика...")
        self.nav_grid = OccupancyGrid(width_m=100.0, height_m=100.0, resolution=0.1, inflation_radius=0.35)
        self.planner = AStarPlanner(self.nav_grid)
        self.robot_x, self.robot_y, self.robot_yaw = 0.0, 0.0, 0.0
        self.last_update_time = self._now_sec()
        self.state = "EXPLORE"
        self.last_map_print_time = self._now_sec()

        # --- Vision ---
        self.orb = cv2.ORB_create(nfeatures=1200)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.face_warp_size = 220
        self.templates = []
        self._load_templates()

        self.create_timer(0.05, self._main_loop)
        self.get_logger().info("Controller started. Режим: Динамические параметры CameraInfo.")

    def _cam_info_callback(self, msg: CameraInfo):
        """Динамически вычисляет FOV на основе матрицы K из топика."""
        fx = msg.k[0]
        fy = msg.k[4]
        if fx > 0 and fy > 0:
            print(msg)
            self.hfov_rad = 2.0 * math.atan(msg.width / (2.0 * fx))
            self.vfov_rad = 2.0 * math.atan(msg.height / (2.0 * fy))
            self.cam_intrinsics_ready = True
        self.latest_cam_info = msg

    def get_expected_object_id(self) -> Optional[int]:
        if self.sequence_index >= len(self.sequence_of_objects): return None
        entry = self.sequence_of_objects[self.sequence_index]
        return int(entry[0]) if isinstance(entry, (tuple, list)) else int(entry)

    def get_expected_object_name(self) -> Optional[str]:
        if self.sequence_index >= len(self.sequence_of_objects): return None
        entry = self.sequence_of_objects[self.sequence_index]
        return str(entry[1]) if isinstance(entry, (tuple, list)) and len(entry) > 1 else None

    def publish_detected_object(self, object_id: int) -> None:
        msg = Int32();
        msg.data = int(object_id)
        self.detected_object_pub.publish(msg)

    def log_detected_object(self, object_id: int, object_name: str, t: float) -> None:
        if object_id in self.detected_objects: return
        self.publish_detected_object(object_id)
        self.detected_objects[object_id] = {
            "name": object_name, "t": round(t, 3), "x": round(self.robot_x, 4),
            "y": round(self.robot_y, 4), "yaw": round(self.robot_yaw, 4),
        }
        self.get_logger().info(f"OBJECT SAVED | id={object_id} | name={object_name}")

    def update_majority_vote(self, detection: dict) -> Optional[dict]:
        if detection is None: return None
        self.class_history.append((detection["id"], detection["name"]))
        ids = [x[0] for x in self.class_history]
        names = {obj_id: obj_name for obj_id, obj_name in self.class_history}
        leader_id, leader_count = Counter(ids).most_common(1)[0]
        return {"id": leader_id, "name": names[leader_id], "count": leader_count}

    def _load_templates(self):
        base_path = '/workspace/aliengo_competition/resources/assets/objects/'
        paths = ['backpack/backpack.jpg', 'bottle/bottle.png', 'chair/chair.png', 'cup/cup.jpg', 'laptop/laptop.png']
        for i, p in enumerate(paths):
            img = cv2.imread(base_path + p, 0)
            if img is not None:
                img = cv2.resize(img, (self.face_warp_size, self.face_warp_size))
                kp, des = self.orb.detectAndCompute(img, None)
                if des is not None:
                    self.templates.append(
                        {"id": i, "name": p, "kp": kp, "des": des, "width": img.shape[1], "height": img.shape[0]})

    def order_quad_points(self, pts: np.ndarray) -> np.ndarray:
        pts = pts.astype(np.float32);
        s = pts.sum(axis=1);
        d = np.diff(pts, axis=1).reshape(-1)
        return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)

    def estimate_bbox_size_m(self, bbox: tuple, depth_m: float) -> tuple:
        if self.latest_rgb is None: return 999.0, 999.0
        x1, y1, x2, y2 = bbox
        h, w = self.latest_rgb.shape[:2]
        # Используем динамический расчет на основе текущего FOV
        width_m = 2.0 * depth_m * math.tan(self.hfov_rad / 2.0) * (max(1.0, x2 - x1) / w)
        height_m = 2.0 * depth_m * math.tan(self.vfov_rad / 2.0) * (max(1.0, y2 - y1) / h)
        return width_m, height_m

    def get_depth_at_rgb_bbox_center(self, bbox: tuple, window: int = 5) -> Optional[float]:
        if self.latest_depth is None: return None
        x1, y1, x2, y2 = bbox
        dh, dw = self.latest_depth.shape[:2]
        rh, rw = (self.latest_rgb.shape[:2] if self.latest_rgb is not None else (dh, dw))
        cx, cy = int(((x1 + x2) / 2) * dw / rw), int(((y1 + y2) / 2) * dh / rh)
        patch = self.latest_depth[max(0, cy - window):min(dh, cy + window + 1),
                max(0, cx - window):min(dw, cx + window + 1)]
        valid = patch[np.isfinite(patch) & (patch > 0.15) & (patch < 8.0)]
        return float(np.median(valid)) if len(valid) > 0 else None

    def get_laser_scan(self) -> Optional[np.ndarray]:
        if self.latest_depth is None: return None
        h, w = self.latest_depth.shape
        raw_row = np.nanmedian(self.latest_depth[int(h * 0.35) - 2: int(h * 0.35) + 3, :], axis=0)
        # Динамический FOV для коррекции дуги
        angles = np.linspace(-self.hfov_rad / 2, self.hfov_rad / 2, w)
        corrected = np.nan_to_num(raw_row / np.cos(angles), posinf=5.0, neginf=0.0)
        corrected[corrected > 4.0] = 5.0;
        corrected[corrected < 0.2] = 5.0
        return corrected

    def detect_object(self) -> Optional[dict]:
        # Вставьте здесь вашу реализацию detect_object из предыдущих версий
        # Она должна использовать detect_cube_face_candidates() и classify_warped_face()
        return None

    def run_user_code(self) -> None:
        now = self._now_sec()
        dt, self.last_update_time = now - self.last_update_time, now
        if dt > 0.5: dt = 0.05

        # 1. Одометрия
        vx, wz = self.latest_base_velocity["vx"], self.latest_base_velocity["wz"]
        self.robot_yaw = (self.robot_yaw + wz * dt + math.pi) % (2 * math.pi) - math.pi
        self.robot_x += vx * math.cos(self.robot_yaw) * dt
        self.robot_y += vx * math.sin(self.robot_yaw) * dt

        # 2. Застревание
        if math.hypot(self.robot_x - self.last_valid_x, self.robot_y - self.last_valid_y) > 0.2:
            self.last_valid_x, self.last_valid_y, self.stuck_start_time = self.robot_x, self.robot_y, now
        if not self.is_recovering and (now - self.stuck_start_time > 10.0):
            self.is_recovering, self.recovery_end_time = True, now + 5.0

        if self.is_recovering:
            if now < self.recovery_end_time:
                self.send_command(-1.5, 0.0, 2.0); return
            else:
                self.is_recovering = False

        # 3. Вижн и Навигация
        scan = self.get_laser_scan()
        dist_L, dist_C, dist_R = 5.0, 5.0, 5.0
        if scan is not None:
            w = len(scan)
            angles = np.linspace(-self.hfov_rad / 2, self.hfov_rad / 2, w)
            for i in range(0, w, 15):
                dist, ang = scan[i], self.robot_yaw + angles[i]
                if dist < 0.3: continue
                for r in np.arange(0.2, min(dist, 4.0), 0.2):
                    self.nav_grid.mark_free(self.robot_x + r * math.cos(ang), self.robot_y + r * math.sin(ang))
                if dist < 4.0: self.nav_grid.mark_obstacle(self.robot_x + dist * math.cos(ang),
                                                           self.robot_y + dist * math.sin(ang))
            dist_L, dist_C, dist_R = np.min(scan[:w // 3]), np.min(scan[w // 3:2 * w // 3]), np.min(scan[2 * w // 3:])

        # 4. Управление (Explore)
        cvx, cvy, cwz = 0.0, 0.0, 0.0
        if dist_C < 1.2:
            cvx, cwz = 0.2, (2.0 if dist_L > dist_R else -2.0)
        elif dist_L < 0.7:
            cvx, cvy, cwz = 1.5, -0.75, -1.0
        elif dist_R < 0.7:
            cvx, cvy, cwz = 1.5, 0.75, 1.0
        else:
            cvx = 1.5

        if now > self.stop_until_time:
            self.send_command(cvx, cvy, cwz)
        else:
            self.send_command(0.0, 0.0, 0.0)

    def send_command(self, vx, vy, wz):
        msg = Twist();
        msg.linear.x, msg.linear.y, msg.angular.z = float(vx), float(vy), float(wz)
        self.cmd_pub.publish(msg)

    # ---------------- CALLBACKS ----------------
    def _vel_callback(self, msg):
        self.latest_base_velocity = {"vx": msg.linear_vel[0], "vy": msg.linear_vel[1], "wz": msg.angular_velocity[2],
                                     "stamp_sec": self._msg_time_to_sec(msg.header.timestamp)}

    def _joint_callback(self, msg):
        self.latest_joint_state = {"names": list(msg.name), "position": list(msg.position),
                                   "name_to_index": {n: i for i, n in enumerate(msg.name)},
                                   "stamp_sec": self._msg_time_to_sec(msg.header.stamp)}

    def _imu_callback(self, msg):
        self.latest_imu = {"wx": msg.angular_vel[0], "wy": msg.angular_vel[1], "wz": msg.angular_vel[2],
                           "stamp_sec": self._msg_time_to_sec(msg.header.timestamp)}

    def _rgb_callback(self, msg):
        self.latest_rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))

    def _depth_callback(self, msg):
        self.latest_depth = np.frombuffer(msg.data, dtype=np.float32).reshape((msg.height, msg.width))

    def _main_loop(self):
        self.run_user_code()

    def _now_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _msg_time_to_sec(self, stamp):
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def save_full_map(self, filename):
        pass


def main(args=None):
    rclpy.init(args=args)
    info_pub = DepthCameraInfoPublisher()
    controller = HLInterfaceController()
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(info_pub);
    executor.add_node(controller)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        controller.send_command(0, 0, 0)
        rclpy.shutdown()


if __name__ == "__main__": main()