import json
import os
import sys
from dataclasses import dataclass

import numpy as np

from avatarposer_realtime.transforms import as_numpy, euler_degrees_to_matrix


WRIST_OFFSET_PRESETS = {
    "none": {
        "left_hand": None,
        "right_hand": None,
    },
    "htc_mr": {
        "left_hand": None,
        "right_hand": None,
    },
    "nolo_openvr_controller": {
        "left_hand": None,
        "right_hand": None,
    },
    "nolo_openvr_controller_mirrored": {
        "left_hand": None,
        "right_hand": None,
    },
    "legacy_nolo_openvr_controller": {
        "left_hand": (-180.0, 90.0, -90.0),
        "right_hand": (-180.0, 90.0, -90.0),
    },
    "legacy_nolo_openvr_controller_mirrored": {
        "left_hand": (-180.0, 90.0, -90.0),
        "right_hand": (180.0, -90.0, 90.0),
    },
}


@dataclass(frozen=True)
class DeviceJointCalibration:
    """Local transform from a tracked device pose to the AvatarPoser training joint pose."""

    position_offset: np.ndarray
    rotation_offset: np.ndarray

    @classmethod
    def identity(cls):
        return cls(
            position_offset=np.zeros(3, dtype=np.float32),
            rotation_offset=np.eye(3, dtype=np.float32),
        )

    @classmethod
    def from_config(cls, config):
        config = config or {}
        position_offset = as_numpy(config.get("position_offset", [0.0, 0.0, 0.0])).reshape(3)
        if "rotation_offset_matrix" in config:
            rotation_offset = as_numpy(config["rotation_offset_matrix"]).reshape(3, 3)
        else:
            rotation_offset = euler_degrees_to_matrix(config.get("rotation_offset_deg", [0.0, 0.0, 0.0]))
        return cls(position_offset=position_offset, rotation_offset=rotation_offset)

    def apply(self, rotation, position):
        rotation = as_numpy(rotation).reshape(3, 3)
        position = as_numpy(position).reshape(3)
        joint_position = position + rotation @ self.position_offset
        joint_rotation = rotation @ self.rotation_offset
        return joint_rotation.astype(np.float32), joint_position.astype(np.float32)


class TrackingCalibration:
    def __init__(self, devices=None, path=None):
        self.devices = devices or {}
        self.path = path

    @classmethod
    def identity(cls):
        return cls()

    @classmethod
    def from_config(cls, config, path=None):
        devices_config = (config or {}).get("devices", {})
        devices = {
            device_name: DeviceJointCalibration.from_config(device_config)
            for device_name, device_config in devices_config.items()
            if device_config is not None
        }
        return cls(devices=devices, path=path)

    def apply(self, device_name, rotation, position):
        calibration = self.devices.get(device_name)
        if calibration is None:
            return as_numpy(rotation).reshape(3, 3), as_numpy(position).reshape(3)
        return calibration.apply(rotation, position)

    def describe(self):
        if not self.devices:
            return "identity"
        names = ", ".join(sorted(self.devices.keys()))
        if self.path:
            return "{} ({})".format(self.path, names)
        return names


def load_tracking_calibration(path=None, stream=None):
    if not path:
        return TrackingCalibration.identity()
    if not os.path.exists(path):
        print(
            "[calibration] tracking calibration file not found, using identity: {}".format(path),
            file=stream or sys.stderr,
        )
        return TrackingCalibration.identity()

    with open(path, "r", encoding="utf-8") as file:
        config = json.load(file)
    calibration = TrackingCalibration.from_config(config, path=path)
    print("[calibration] loaded tracking calibration: {}".format(calibration.describe()), file=stream or sys.stderr)
    return calibration


def wrist_offset_preset_names():
    return tuple(WRIST_OFFSET_PRESETS.keys())


def resolve_wrist_offset_degrees(preset_name="none", left_hand=None, right_hand=None):
    if preset_name not in WRIST_OFFSET_PRESETS:
        raise ValueError("Unknown wrist offset preset: {}".format(preset_name))

    preset = WRIST_OFFSET_PRESETS[preset_name]
    resolved_left = left_hand if left_hand is not None else preset["left_hand"]
    resolved_right = right_hand if right_hand is not None else preset["right_hand"]
    return {
        "left_hand": resolved_left,
        "right_hand": resolved_right,
    }
