import numpy as np

from avatarposer_realtime.transforms import as_numpy, matrix_to_sixd, sixd_to_matrix
from avatarposer_realtime.types import TrackingFeatureFrame, TrackingFrame


class TrackingFeatureBuilder:
    """Build AvatarPoser's 54D feature from head/left/right tracking poses."""

    def __init__(self):
        self.prev_rotations = None
        self.prev_positions = None

    def update(self, frame: TrackingFrame) -> TrackingFeatureFrame:
        rotations = np.stack(
            [
                frame.head.rotation,
                frame.left_hand.rotation,
                frame.right_hand.rotation,
            ],
            axis=0,
        ).astype(np.float32)
        positions = np.stack(
            [
                frame.head.position,
                frame.left_hand.position,
                frame.right_hand.position,
            ],
            axis=0,
        ).astype(np.float32)

        rotation_6d = matrix_to_sixd(rotations)
        if self.prev_rotations is None:
            rotation_velocity = np.repeat(np.eye(3, dtype=np.float32)[None], 3, axis=0)
            position_velocity = np.zeros_like(positions)
        else:
            rotation_velocity = np.matmul(np.linalg.inv(self.prev_rotations), rotations)
            position_velocity = positions - self.prev_positions

        rotation_velocity_6d = matrix_to_sixd(rotation_velocity)
        self.prev_rotations = rotations.copy()
        self.prev_positions = positions.copy()

        feature = np.concatenate(
            [
                rotation_6d.reshape(-1),
                rotation_velocity_6d.reshape(-1),
                positions.reshape(-1),
                position_velocity.reshape(-1),
            ],
            axis=0,
        ).astype(np.float32)

        head_transform = np.eye(4, dtype=np.float32)
        head_transform[:3, :3] = rotations[0]
        head_transform[:3, 3] = positions[0]
        return TrackingFeatureFrame(
            frame_index=frame.frame_index,
            feature=feature,
            head_transform=head_transform,
            timestamp=frame.timestamp,
        )


def feature_to_head_transform(feature):
    feature = as_numpy(feature).reshape(54)
    head_rotation = sixd_to_matrix(feature[:6])[0]
    head_position = feature[36:39]
    head_transform = np.eye(4, dtype=np.float32)
    head_transform[:3, :3] = head_rotation
    head_transform[:3, 3] = head_position
    return head_transform

