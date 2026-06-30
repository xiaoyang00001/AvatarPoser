import glob
import json
import os
import pickle
import socket
import sys
from typing import Iterator, Optional

from avatarposer_realtime.calibration import (
    TrackingCalibration,
    load_tracking_calibration,
    resolve_wrist_offset_degrees,
)
from avatarposer_realtime.feature_builder import TrackingFeatureBuilder, feature_to_head_transform
from avatarposer_realtime.transforms import (
    as_numpy,
    euler_degrees_to_matrix,
    parse_position,
    parse_rotation,
    transform_position,
    transform_rotation,
)
from avatarposer_realtime.types import DEVICE_ALIASES, DevicePose, TrackingFeatureFrame, TrackingFrame


DEVICE_ALIASES_BY_NAME = {aliases[0]: aliases for aliases in DEVICE_ALIASES}


class InvalidTrackingFrame(ValueError):
    pass


def find_device_pose(record, aliases):
    devices = record.get("devices", record)
    for alias in aliases:
        if alias in devices:
            return devices[alias]
    raise KeyError("Missing tracking device. Expected one of: {}".format(", ".join(aliases)))


def tracking_frame_from_record(
    frame_index,
    record,
    coordinate_system="identity",
    orientation_transform="pose",
    rotation_offsets=None,
    tracking_calibration=None,
    swap_hands=False,
) -> TrackingFrame:
    rotation_offsets = rotation_offsets or {}
    tracking_calibration = tracking_calibration or TrackingCalibration.identity()
    poses = []
    for aliases in DEVICE_ALIASES:
        device_name = aliases[0]
        source_aliases = aliases
        if swap_hands and device_name == "left_hand":
            source_aliases = DEVICE_ALIASES_BY_NAME["right_hand"]
        elif swap_hands and device_name == "right_hand":
            source_aliases = DEVICE_ALIASES_BY_NAME["left_hand"]

        pose_record = find_device_pose(record, source_aliases)
        try:
            rotation = parse_rotation(pose_record)
            position = parse_position(pose_record)
            rotation = transform_rotation(rotation, coordinate_system, mode=orientation_transform)
            position = transform_position(position, coordinate_system)
            rotation, position = tracking_calibration.apply(device_name, rotation, position)
            if device_name in rotation_offsets:
                rotation = rotation @ rotation_offsets[device_name]
            pose = DevicePose(rotation, position)
        except ValueError as exc:
            raise InvalidTrackingFrame("Invalid {} pose: {}".format(device_name, exc)) from exc

        poses.append(pose)

    return TrackingFrame(
        frame_index=int(record.get("frame", record.get("frame_index", frame_index))),
        head=poses[0],
        left_hand=poses[1],
        right_hand=poses[2],
        timestamp=float(record.get("timestamp", 0.0)),
    )


def feature_frame_from_record(
    frame_index,
    record,
    builder,
    coordinate_system="identity",
    orientation_transform="pose",
    rotation_offsets=None,
    tracking_calibration=None,
    swap_hands=False,
) -> TrackingFeatureFrame:
    if "feature" in record:
        feature = as_numpy(record["feature"]).reshape(54)
        head_transform = record.get("head_transform")
        if head_transform is None:
            head_transform = feature_to_head_transform(feature)
        else:
            head_transform = as_numpy(head_transform).reshape(4, 4)

        return TrackingFeatureFrame(
            frame_index=int(record.get("frame", record.get("frame_index", frame_index))),
            feature=feature,
            head_transform=head_transform,
            timestamp=float(record.get("timestamp", 0.0)),
        )

    return builder.update(
        tracking_frame_from_record(
            frame_index,
            record,
            coordinate_system,
            orientation_transform,
            rotation_offsets,
            tracking_calibration,
            swap_hands,
        )
    )


class PklReplaySource:
    def __init__(self, path: str):
        self.path = path

    def __iter__(self) -> Iterator[TrackingFeatureFrame]:
        with open(self.path, "rb") as f:
            data = pickle.load(f)
        features = as_numpy(data["hmd_position_global_full_gt_list"])
        transforms = as_numpy(data["head_global_trans_list"])
        frame_count = min(len(features), len(transforms))
        for frame_index in range(frame_count):
            yield TrackingFeatureFrame(frame_index, features[frame_index], transforms[frame_index])


class JsonLineTrackingSource:
    def __init__(
        self,
        stream=None,
        coordinate_system="identity",
        orientation_transform="pose",
        rotation_offsets=None,
        tracking_calibration=None,
        swap_hands=False,
    ):
        self.stream = stream or sys.stdin
        self.builder = TrackingFeatureBuilder()
        self.coordinate_system = coordinate_system
        self.orientation_transform = orientation_transform
        self.rotation_offsets = rotation_offsets or {}
        self.tracking_calibration = tracking_calibration or TrackingCalibration.identity()
        self.swap_hands = bool(swap_hands)

    def __iter__(self) -> Iterator[TrackingFeatureFrame]:
        for frame_index, line in enumerate(self.stream):
            line = line.strip()
            if not line:
                continue
            yield feature_frame_from_record(
                frame_index,
                json.loads(line),
                self.builder,
                self.coordinate_system,
                self.orientation_transform,
                self.rotation_offsets,
                self.tracking_calibration,
                self.swap_hands,
            )


class NoloUdpTrackingSource:
    """Receive JSON tracking packets exported by NOLO_Link_Driver."""

    def __init__(
        self,
        host="127.0.0.1",
        port=39001,
        buffer_size=8192,
        timeout: Optional[float] = None,
        latest_only=True,
        receive_buffer_size=65536,
        coordinate_system="htc_mr",
        orientation_transform="pose",
        rotation_offsets=None,
        tracking_calibration=None,
        swap_hands=False,
    ):
        self.host = host
        self.port = int(port)
        self.buffer_size = buffer_size
        self.timeout = timeout
        self.latest_only = latest_only
        self.receive_buffer_size = receive_buffer_size
        self.coordinate_system = coordinate_system
        self.orientation_transform = orientation_transform
        self.rotation_offsets = rotation_offsets or {}
        self.tracking_calibration = tracking_calibration or TrackingCalibration.identity()
        self.swap_hands = bool(swap_hands)
        self.builder = TrackingFeatureBuilder()

    def receive_payload(self, sock):
        payload, _addr = sock.recvfrom(self.buffer_size)
        dropped_packets = 0
        if not self.latest_only:
            return payload, dropped_packets

        sock.setblocking(False)
        try:
            while True:
                try:
                    payload, _addr = sock.recvfrom(self.buffer_size)
                    dropped_packets += 1
                except BlockingIOError:
                    return payload, dropped_packets
        finally:
            sock.setblocking(True)
            if self.timeout is not None:
                sock.settimeout(self.timeout)

    def __iter__(self) -> Iterator[TrackingFeatureFrame]:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, int(self.receive_buffer_size))
        sock.bind((self.host, self.port))
        if self.timeout is not None:
            sock.settimeout(self.timeout)
        try:
            frame_index = 0
            skipped_frames = 0
            dropped_packets_total = 0
            while True:
                payload, dropped_packets = self.receive_payload(sock)
                dropped_packets_total += dropped_packets
                if dropped_packets and (dropped_packets_total == dropped_packets or dropped_packets_total % 300 == 0):
                    print(
                        "[tracking] dropped {} stale UDP packet(s) to keep realtime latency low.".format(
                            dropped_packets_total
                        ),
                        file=sys.stderr,
                    )
                try:
                    record = json.loads(payload.decode("utf-8"))
                    feature_frame = feature_frame_from_record(
                        frame_index,
                        record,
                        self.builder,
                        self.coordinate_system,
                        self.orientation_transform,
                        self.rotation_offsets,
                        self.tracking_calibration,
                        self.swap_hands,
                    )
                except (json.JSONDecodeError, KeyError, InvalidTrackingFrame) as exc:
                    skipped_frames += 1
                    if skipped_frames == 1 or skipped_frames % 60 == 0:
                        print(
                            "[tracking] waiting for valid NOLO head/hand poses; skipped {} packet(s): {}".format(
                                skipped_frames, exc
                            ),
                            file=sys.stderr,
                        )
                    continue

                yield feature_frame
                frame_index += 1
        finally:
            sock.close()


def default_pkl_source():
    matches = sorted(glob.glob(os.path.join("data_fps60", "*", "test", "*.pkl")))
    if not matches:
        raise FileNotFoundError("No test pkl found under data_fps60/*/test/*.pkl")
    return matches[0]


def create_source(
    source,
    udp_host="127.0.0.1",
    udp_port=39001,
    latest_only=True,
    receive_buffer_size=65536,
    coordinate_system="htc_mr",
    orientation_transform="pose",
    wrist_offset_preset="none",
    left_wrist_offset_degrees=None,
    right_wrist_offset_degrees=None,
    tracking_calibration_path=None,
    swap_hands=False,
):
    rotation_offsets = {}
    wrist_offset_degrees = resolve_wrist_offset_degrees(
        wrist_offset_preset,
        left_hand=left_wrist_offset_degrees,
        right_hand=right_wrist_offset_degrees,
    )
    if wrist_offset_degrees["left_hand"] is not None:
        rotation_offsets["left_hand"] = euler_degrees_to_matrix(wrist_offset_degrees["left_hand"])
    if wrist_offset_degrees["right_hand"] is not None:
        rotation_offsets["right_hand"] = euler_degrees_to_matrix(wrist_offset_degrees["right_hand"])
    tracking_calibration = load_tracking_calibration(tracking_calibration_path)

    if source == "auto":
        return PklReplaySource(default_pkl_source())
    if source == "stdin":
        return JsonLineTrackingSource(
            coordinate_system=coordinate_system,
            orientation_transform=orientation_transform,
            rotation_offsets=rotation_offsets,
            tracking_calibration=tracking_calibration,
            swap_hands=swap_hands,
        )
    if source == "udp":
        return NoloUdpTrackingSource(
            udp_host,
            udp_port,
            latest_only=latest_only,
            receive_buffer_size=receive_buffer_size,
            coordinate_system=coordinate_system,
            orientation_transform=orientation_transform,
            rotation_offsets=rotation_offsets,
            tracking_calibration=tracking_calibration,
            swap_hands=swap_hands,
        )
    return PklReplaySource(source)
