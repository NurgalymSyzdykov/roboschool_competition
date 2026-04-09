from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class VelocityCommand:
    vx: float = 0.0
    vy: float = 0.0
    vw: float = 0.0


@dataclass
class JointState:
    names: Tuple[str, ...]
    positions: np.ndarray
    velocities: np.ndarray


@dataclass
class ImuState:
    angular_velocity_xyz: np.ndarray


@dataclass
class CameraState:
    rgb: Optional[np.ndarray]
    depth: Optional[np.ndarray]


@dataclass
class RobotState:
    step_index: int
    sim_time_s: float
    dt: float
    joints: JointState
    imu: ImuState
    base_linear_velocity_xyz: np.ndarray
    base_angular_velocity_xyz: np.ndarray
    camera: CameraState

    @property
    def q(self) -> np.ndarray:
        return self.joints.positions

    @property
    def q_dot(self) -> np.ndarray:
        return self.joints.velocities

    @property
    def joint_names(self) -> Tuple[str, ...]:
        return self.joints.names

    @property
    def linear_velocity_xyz(self) -> np.ndarray:
        return self.base_linear_velocity_xyz
