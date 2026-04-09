from __future__ import annotations

import math

import numpy as np

from aliengo_competition.robot_interface.base import AliengoRobotInterface
from aliengo_competition.robot_interface.types import CameraState


def _unwrap_env_from_robot(robot: AliengoRobotInterface):
    env = getattr(robot, "env", None)
    while env is not None and hasattr(env, "env") and getattr(env, "env") is not env:
        env = env.env
    return env


def _infer_control_dt(robot: AliengoRobotInterface, fallback_dt: float = 0.02) -> float:
    env = _unwrap_env_from_robot(robot)
    dt = getattr(env, "dt", None) if env is not None else None
    try:
        dt_value = float(dt)
        if dt_value > 0.0:
            return dt_value
    except (TypeError, ValueError):
        pass
    return float(fallback_dt)


class _CameraRenderer:
    def __init__(self, enabled: bool, depth_max_m: float):
        self.enabled = bool(enabled)
        self.depth_max_m = max(float(depth_max_m), 0.1)
        self._window_name = "Front Camera (Intel RealSense D435-like)"
        self._cv2 = None
        self._active = False
        if not self.enabled:
            return
        try:
            import cv2
        except Exception as exc:
            print(f"Camera rendering disabled: failed to import cv2 ({exc})")
            self.enabled = False
            return
        self._cv2 = cv2
        self._cv2.namedWindow(self._window_name, self._cv2.WINDOW_NORMAL)
        self._active = True

    def show(self, camera: CameraState) -> None:
        if not self._active or not isinstance(camera, CameraState):
            return
        image = camera.rgb
        depth = camera.depth
        if image is None or depth is None:
            return

        rgb = np.asarray(image)
        depth_m = np.asarray(depth, dtype=np.float32)
        if rgb.ndim != 3 or rgb.shape[2] < 3 or depth_m.ndim != 2:
            return
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        rgb = rgb[..., :3]
        depth_m = np.nan_to_num(depth_m, nan=0.0, posinf=self.depth_max_m, neginf=0.0)
        depth_m = np.clip(depth_m, 0.0, self.depth_max_m)
        depth_u8 = (depth_m * (255.0 / self.depth_max_m)).astype(np.uint8)

        cv2 = self._cv2
        depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
        depth_color = cv2.resize(depth_color, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        view = np.concatenate((rgb_bgr, depth_color), axis=1)

        cv2.putText(view, "RGB", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            view,
            f"Depth 0..{self.depth_max_m:.1f}m",
            (rgb.shape[1] + 10, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(self._window_name, view)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            self.close()

    def close(self) -> None:
        if not self._active or self._cv2 is None:
            return
        self._cv2.destroyWindow(self._window_name)
        self._active = False


def run(
    robot: AliengoRobotInterface,
    steps: int = 1000,
    render_camera: bool = False,
    camera_depth_max_m: float = 10.0,
) -> None:
    robot.reset()
    camera_renderer = _CameraRenderer(enabled=render_camera, depth_max_m=camera_depth_max_m)
    control_dt = _infer_control_dt(robot, fallback_dt=0.02)
    requested_steps = max(int(steps), 1)
    nominal_dt = 0.02
    target_duration_s = requested_steps * nominal_dt
    total_steps = max(int(round(target_duration_s / control_dt)), 1)
    print(
        f"[Controller] dt={control_dt:.4f}s, requested_steps={requested_steps}, "
        f"effective_steps={total_steps}"
    )

    # User-editable blocks in this file:
    # 1. USER PARAMETERS START / END
    # 2. USER CONTROL LOGIC START / END

    # ================= USER PARAMETERS START =================
    # Tune these values to change the demo behavior. Time-based settings are
    # converted with env.dt, so behavior is stable when sim step changes.
    warmup_s = 0.4
    ramp_s = 1.2
    trajectory_period_s = 8.0
    forward_speed_mean = 0.40
    forward_speed_amp = 0.35
    lateral_speed_amp = 0.22
    yaw_rate_amp = 0.75
    yaw_rate_damping = 0.55
    ang_vel_scale = 0.25
    # ================== USER PARAMETERS END ==================

    segment_start_t = 0.0

    try:
        for step_index in range(total_steps):
            state = robot.get_state()
            camera_renderer.show(state.camera)
            omega_z = float(state.imu.angular_velocity_xyz[2]) / ang_vel_scale

            # ================= USER CONTROL LOGIC START =================
            # This block is the intended place for participant logic.
            # You can:
            # - read state.q / state.q_dot / state.imu / state.linear_velocity_xyz / state.camera
            # - compute desired vx, vy, vw
            #
            # Example below:
            # - smooth warmup
            # - continuous figure-eight in velocity space
            # - yaw-rate command combines feed-forward turn and damping
            sim_t = state.sim_time_s
            local_t = max(sim_t - segment_start_t, 0.0)
            if local_t < warmup_s:
                vx = 0.0
                vy = 0.0
                vw = 0.0
            else:
                motion_t = local_t - warmup_s
                phase = 2.0 * math.pi * motion_t / max(trajectory_period_s, control_dt)
                ramp = min(motion_t / max(ramp_s, control_dt), 1.0)

                vx = ramp * (forward_speed_mean + forward_speed_amp * math.cos(phase))
                vy = ramp * (lateral_speed_amp * math.sin(2.0 * phase))
                yaw_ff = yaw_rate_amp * math.sin(phase + math.pi / 4.0)
                vw = ramp * (yaw_ff - yaw_rate_damping * omega_z)
                vw = max(min(vw, 1.0), -1.0)
            # ================== USER CONTROL LOGIC END ==================

            robot.set_speed(vx, vy, vw)
            robot.step()
            if robot.is_fallen():
                robot.stop()
                robot.reset()
                segment_start_t = (step_index + 1) * control_dt
                continue
    finally:
        camera_renderer.close()
        robot.stop()
