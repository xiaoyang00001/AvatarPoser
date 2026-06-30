from dataclasses import dataclass
from typing import Dict, List

import numpy as np


JOINT_NAMES: List[str] = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]

DEVICE_ALIASES = [
    ("head", "hmd"),
    ("left_hand", "left", "left_controller"),
    ("right_hand", "right", "right_controller"),
]


@dataclass
class DevicePose:
    rotation: np.ndarray
    position: np.ndarray


@dataclass
class TrackingFrame:
    frame_index: int
    head: DevicePose
    left_hand: DevicePose
    right_hand: DevicePose
    timestamp: float = 0.0


@dataclass
class TrackingFeatureFrame:
    frame_index: int
    feature: np.ndarray
    head_transform: np.ndarray
    timestamp: float = 0.0


@dataclass
class AvatarPoseResult:
    frame_index: int
    joints: np.ndarray
    pose_body: np.ndarray
    root_orient: np.ndarray
    translation: np.ndarray
    body: object

    def joints_by_name(self) -> Dict[str, List[float]]:
        return {
            name: [float(value) for value in self.joints[index]]
            for index, name in enumerate(JOINT_NAMES)
        }

