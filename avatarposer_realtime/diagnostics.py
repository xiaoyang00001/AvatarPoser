import sys

import numpy as np

from avatarposer_realtime.transforms import matrix_to_quaternion_wxyz, sixd_to_matrix
from avatarposer_realtime.types import JOINT_NAMES


class TrackingDebugger:
    def __init__(self, every=0, stream=None):
        self.every = int(every)
        self.stream = stream or sys.stderr

    def update(self, feature_frame):
        if self.every <= 0 or feature_frame.frame_index % self.every != 0:
            return

        positions = np.asarray(feature_frame.feature[36:45], dtype=np.float32).reshape(3, 3)
        head, left_hand, right_hand = positions
        hand_distance = float(np.linalg.norm(left_hand - right_hand))
        head_left_distance = float(np.linalg.norm(head - left_hand))
        head_right_distance = float(np.linalg.norm(head - right_hand))
        print(
            "[tracking-debug] frame={} head={} left={} right={} "
            "head_y={:.3f} head_z={:.3f} hand_dist={:.3f} head_left={:.3f} head_right={:.3f}".format(
                feature_frame.frame_index,
                np.round(head, 3).tolist(),
                np.round(left_hand, 3).tolist(),
                np.round(right_hand, 3).tolist(),
                float(head[1]),
                float(head[2]),
                hand_distance,
                head_left_distance,
                head_right_distance,
            ),
            file=self.stream,
        )


class RotationAxesDebugger:
    def __init__(self, every=0, stream=None):
        self.every = int(every)
        self.stream = stream or sys.stderr
        self.device_names = ("head", "left_hand", "right_hand")

    def update(self, feature_frame):
        if self.every <= 0 or feature_frame.frame_index % self.every != 0:
            return

        rotations = sixd_to_matrix(np.asarray(feature_frame.feature[:18], dtype=np.float32).reshape(3, 6))
        print("[rotation-debug] frame={}".format(feature_frame.frame_index), file=self.stream)
        for device_name, rotation in zip(self.device_names, rotations):
            det = float(np.linalg.det(rotation))
            orth_error = float(np.linalg.norm(rotation.T @ rotation - np.eye(3, dtype=np.float32)))
            x_axis = np.round(rotation[:, 0], 3).tolist()
            y_axis = np.round(rotation[:, 1], 3).tolist()
            z_axis = np.round(rotation[:, 2], 3).tolist()
            quat_wxyz = np.round(matrix_to_quaternion_wxyz(rotation), 4).tolist()
            print(
                "  {} det={:.3f} orth_err={:.5f} quat_wxyz={} local_x={} local_y={} local_z={}".format(
                    device_name,
                    det,
                    orth_error,
                    quat_wxyz,
                    x_axis,
                    y_axis,
                    z_axis,
                ),
                file=self.stream,
            )


class RootPoseDebugger:
    def __init__(self, every=0, stream=None):
        self.every = int(every)
        self.stream = stream or sys.stderr
        self.joint_indices = {name: index for index, name in enumerate(JOINT_NAMES)}

    def update(self, result):
        if self.every <= 0 or result.frame_index % self.every != 0:
            return

        joints = np.asarray(result.joints, dtype=np.float32)
        pelvis = joints[self.joint_indices["pelvis"]]
        head = joints[self.joint_indices["head"]]
        left_wrist = joints[self.joint_indices["left_wrist"]]
        right_wrist = joints[self.joint_indices["right_wrist"]]
        print(
            "[root-debug] frame={} root_orient_aa={} trans={} pelvis_y={:.3f} pelvis_z={:.3f} head_y={:.3f} head_z={:.3f} left_wrist_y={:.3f} left_wrist_z={:.3f} right_wrist_y={:.3f} right_wrist_z={:.3f}".format(
                result.frame_index,
                np.round(result.root_orient, 4).tolist(),
                np.round(result.translation, 3).tolist(),
                float(pelvis[1]),
                float(pelvis[2]),
                float(head[1]),
                float(head[2]),
                float(left_wrist[1]),
                float(left_wrist[2]),
                float(right_wrist[1]),
                float(right_wrist[2]),
            ),
            file=self.stream,
        )
