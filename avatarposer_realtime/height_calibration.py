import sys
from dataclasses import replace

import numpy as np


AXIS_TO_INDEX = {"x": 0, "y": 1, "z": 2}


class HeightCalibrator:
    """Keyboard-triggered head-height correction for live sparse tracking."""

    def __init__(self, source_axis="z", model_axis="z", mode="lock", standing_height=1.5, stream=None):
        if source_axis not in AXIS_TO_INDEX:
            raise ValueError("Unknown source height axis: {}".format(source_axis))
        if model_axis not in AXIS_TO_INDEX:
            raise ValueError("Unknown model height axis: {}".format(model_axis))
        if mode not in ("lock", "copy"):
            raise ValueError("Unknown height calibration mode: {}".format(mode))

        self.source_axis = source_axis
        self.model_axis = model_axis
        self.source_index = AXIS_TO_INDEX[source_axis]
        self.model_index = AXIS_TO_INDEX[model_axis]
        self.mode = mode
        self.standing_height = float(standing_height) if standing_height is not None else 0.0
        self.stream = stream or sys.stderr
        self.pending = False
        self.active = False
        self.locked_height = None

    def request_calibration(self):
        self.pending = True

    def apply(self, feature_frame):
        feature = np.asarray(feature_frame.feature, dtype=np.float32).copy()
        head_transform = np.asarray(feature_frame.head_transform, dtype=np.float32).copy()

        positions = feature[36:45].reshape(3, 3).copy()
        velocities = feature[45:54].reshape(3, 3).copy()
        source_height = float(positions[0, self.source_index])
        model_height = float(positions[0, self.model_index])

        if self.pending:
            if self.mode == "lock":
                self.locked_height = self.standing_height if self.standing_height > 0.0 else model_height
            else:
                self.locked_height = source_height
            self.active = True
            self.pending = False
            print(
                "[height-calibration] calibrated model {} height = {:.4f}m; observed model {} = {:.4f}m; source {} = {:.4f}m; mode={}".format(
                    self.model_axis,
                    self.locked_height,
                    self.model_axis,
                    model_height,
                    self.source_axis,
                    source_height,
                    self.mode,
                ),
                file=self.stream,
            )

        if not self.active:
            return feature_frame

        target_height = source_height if self.mode == "copy" else self.locked_height
        delta = float(target_height) - float(positions[0, self.model_index])
        positions[:, self.model_index] += delta
        velocities[:, self.model_index] = 0.0

        feature[36:45] = positions.reshape(-1)
        feature[45:54] = velocities.reshape(-1)
        head_transform[:3, 3] = positions[0]

        return replace(
            feature_frame,
            feature=feature,
            head_transform=head_transform,
        )
