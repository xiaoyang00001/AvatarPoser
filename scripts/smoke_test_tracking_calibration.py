import os
import sys

import numpy as np


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from avatarposer_realtime.runtime_env import configure_runtime_environment

configure_runtime_environment()

from avatarposer_realtime.calibration import DeviceJointCalibration, TrackingCalibration
from avatarposer_realtime.height_calibration import HeightCalibrator
from avatarposer_realtime.sources import tracking_frame_from_record
from avatarposer_realtime.transforms import euler_degrees_to_matrix
from avatarposer_realtime.types import TrackingFeatureFrame


def assert_close(name, actual, expected, atol=1e-5):
    if not np.allclose(actual, expected, atol=atol):
        raise AssertionError("{} actual={} expected={}".format(name, actual.tolist(), expected.tolist()))


def main():
    calibration = TrackingCalibration(
        devices={
            "head": DeviceJointCalibration(
                position_offset=np.asarray([0.1, -0.2, 0.3], dtype=np.float32),
                rotation_offset=euler_degrees_to_matrix([0.0, 0.0, 90.0]),
            )
        }
    )
    record = {
        "frame": 1,
        "head": {"position": [1.0, 2.0, 3.0], "quaternion": [0.0, 0.0, 0.0, 1.0]},
        "left_hand": {"position": [4.0, 5.0, 6.0], "quaternion": [0.0, 0.0, 0.0, 1.0]},
        "right_hand": {"position": [7.0, 8.0, 9.0], "quaternion": [0.0, 0.0, 0.0, 1.0]},
    }
    frame = tracking_frame_from_record(
        0,
        record,
        coordinate_system="identity",
        orientation_transform="pose",
        tracking_calibration=calibration,
    )
    assert_close("head position", frame.head.position, np.asarray([1.1, 1.8, 3.3], dtype=np.float32))
    assert_close("head rotation", frame.head.rotation, euler_degrees_to_matrix([0.0, 0.0, 90.0]))
    assert_close("left hand unchanged", frame.left_hand.position, np.asarray([4.0, 5.0, 6.0], dtype=np.float32))

    swapped = tracking_frame_from_record(
        0,
        record,
        coordinate_system="identity",
        orientation_transform="pose",
        swap_hands=True,
    )
    assert_close("swapped left hand", swapped.left_hand.position, np.asarray([7.0, 8.0, 9.0], dtype=np.float32))
    assert_close("swapped right hand", swapped.right_hand.position, np.asarray([4.0, 5.0, 6.0], dtype=np.float32))

    feature = np.zeros(54, dtype=np.float32)
    positions = np.asarray(
        [
            [0.1, 0.4, 1.7],
            [0.2, 0.5, 1.2],
            [0.3, 0.6, 1.3],
        ],
        dtype=np.float32,
    )
    velocities = np.asarray(
        [
            [0.0, 0.01, 0.02],
            [0.0, 0.03, 0.04],
            [0.0, 0.05, 0.06],
        ],
        dtype=np.float32,
    )
    feature[36:45] = positions.reshape(-1)
    feature[45:54] = velocities.reshape(-1)
    head_transform = np.eye(4, dtype=np.float32)
    head_transform[:3, 3] = positions[0]
    calibrator = HeightCalibrator(source_axis="y", model_axis="z", mode="lock", standing_height=1.8)
    calibrator.request_calibration()
    calibrated = calibrator.apply(TrackingFeatureFrame(1, feature, head_transform))
    calibrated_positions = calibrated.feature[36:45].reshape(3, 3)
    calibrated_velocities = calibrated.feature[45:54].reshape(3, 3)
    assert_close("height calibrated head", calibrated_positions[0], np.asarray([0.1, 0.4, 1.8], dtype=np.float32))
    assert_close("height calibrated left", calibrated_positions[1], np.asarray([0.2, 0.5, 1.3], dtype=np.float32))
    assert_close("height calibrated right", calibrated_positions[2], np.asarray([0.3, 0.6, 1.4], dtype=np.float32))
    assert_close("height calibrated head transform", calibrated.head_transform[:3, 3], calibrated_positions[0])
    assert_close("height calibrated vertical velocity", calibrated_velocities[:, 2], np.zeros(3, dtype=np.float32))

    next_feature = feature.copy()
    next_positions = positions.copy()
    next_positions[:, 2] -= 0.4
    next_feature[36:45] = next_positions.reshape(-1)
    next_head_transform = head_transform.copy()
    next_head_transform[:3, 3] = next_positions[0]
    locked = calibrator.apply(TrackingFeatureFrame(2, next_feature, next_head_transform))
    locked_positions = locked.feature[36:45].reshape(3, 3)
    assert_close("height locked head", locked_positions[0], np.asarray([0.1, 0.4, 1.8], dtype=np.float32))
    assert_close("height locked left", locked_positions[1], np.asarray([0.2, 0.5, 1.3], dtype=np.float32))
    assert_close("height locked right", locked_positions[2], np.asarray([0.3, 0.6, 1.4], dtype=np.float32))
    print("tracking calibration smoke test passed")


if __name__ == "__main__":
    main()
