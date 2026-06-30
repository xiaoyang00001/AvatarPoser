import argparse
import os
import sys
import time
from collections import deque

import numpy as np

from avatarposer_realtime.calibration import wrist_offset_preset_names
from avatarposer_realtime.diagnostics import RootPoseDebugger, RotationAxesDebugger, TrackingDebugger
from avatarposer_realtime.height_calibration import HeightCalibrator
from avatarposer_realtime.model_runtime import AvatarPoserRuntime
from avatarposer_realtime.sources import create_source
from avatarposer_realtime.transforms import coordinate_transform_names
from avatarposer_realtime.visualizers import JointPrinter, JsonlExporter, create_body_visualizer


def parse_degrees(value):
    values = [float(item.strip()) for item in value.split(",")]
    if len(values) != 3:
        raise argparse.ArgumentTypeError("Expected three comma-separated degree values, e.g. 0,0,90")
    return values


class RealtimeAvatarPoserApp:
    def __init__(self, args):
        self.args = args
        self.runtime = AvatarPoserRuntime(
            args.opt,
            checkpoint=args.checkpoint,
            root_orientation_mode=args.root_orientation_mode,
            upright_constraint=args.upright_constraint,
            head_yaw_axis=args.head_yaw_axis,
            head_yaw_offset_degrees=args.head_yaw_offset_deg,
        )
        self.history = deque(maxlen=self.runtime.window_size)
        self.source = create_source(
            args.source,
            udp_host=args.udp_host,
            udp_port=args.udp_port,
            latest_only=not args.no_drop_stale_udp,
            receive_buffer_size=args.udp_receive_buffer,
            coordinate_system=args.tracking_coordinates,
            orientation_transform=args.orientation_transform,
            wrist_offset_preset=args.wrist_offset_preset,
            left_wrist_offset_degrees=args.left_wrist_offset_deg,
            right_wrist_offset_degrees=args.right_wrist_offset_deg,
            tracking_calibration_path=args.tracking_calibration,
            swap_hands=args.swap_hands,
        )
        self.printer = JointPrinter(every=args.print_every)
        self.tracking_debugger = TrackingDebugger(every=args.debug_tracking_every)
        self.rotation_debugger = RotationAxesDebugger(every=args.debug_rotation_every)
        self.root_debugger = RootPoseDebugger(every=args.debug_root_every)
        self.height_calibrator = HeightCalibrator(
            source_axis=args.height_source_axis,
            model_axis=args.height_model_axis,
            mode=args.height_calibration_mode,
            standing_height=args.standing_head_height,
        )
        self.body_visualizer = None
        self.last_visualize_time = 0.0
        self.frame_count = 0
        self.last_stats_time = time.perf_counter()
        if args.visualize:
            self.body_visualizer = create_body_visualizer(
                self.runtime.body_model,
                width=args.width,
                height=args.height,
                backend=args.visualizer,
                mesh_faces=args.mesh_faces,
                smoothing=args.visual_smoothing,
                camera_mode=args.visual_camera_mode,
            )

    def poll_visualizer_controls(self):
        if not self.body_visualizer:
            return
        poll_events = getattr(self.body_visualizer, "poll_events", None)
        if poll_events:
            poll_events()
        consume_height_request = getattr(self.body_visualizer, "consume_height_calibration_request", None)
        if not consume_height_request:
            return
        while consume_height_request():
            self.height_calibrator.request_calibration()

    def run(self):
        sleep_seconds = 1.0 / self.args.fps if self.args.fps and self.args.fps > 0 else 0.0
        visualize_interval = 1.0 / self.args.visualize_fps if self.args.visualize_fps > 0 else 0.0
        exporter_context = JsonlExporter(self.args.export_jsonl) if self.args.export_jsonl else nullcontext()
        with exporter_context as exporter:
            for feature_frame in self.source:
                frame_start_time = time.perf_counter()
                self.poll_visualizer_controls()
                feature_frame = self.height_calibrator.apply(feature_frame)
                self.tracking_debugger.update(feature_frame)
                self.rotation_debugger.update(feature_frame)
                self.history.append(feature_frame.feature)
                result = self.runtime.predict(
                    feature_frame.frame_index,
                    np.stack(self.history, axis=0),
                    feature_frame.head_transform,
                )

                self.printer.update(result)
                self.root_debugger.update(result)
                if exporter:
                    exporter.update(result)
                if self.body_visualizer:
                    now = time.perf_counter()
                    if visualize_interval <= 0.0 or now - self.last_visualize_time >= visualize_interval:
                        self.body_visualizer.update(result)
                        self.last_visualize_time = now

                self.frame_count += 1
                if self.args.stats_every > 0 and self.frame_count % self.args.stats_every == 0:
                    now = time.perf_counter()
                    elapsed = now - self.last_stats_time
                    if elapsed > 0:
                        fps = self.args.stats_every / elapsed
                        frame_ms = (now - frame_start_time) * 1000.0
                        print(
                            "[stats] processed_fps={:.1f}, last_frame_ms={:.1f}, input_frame={}".format(
                                fps, frame_ms, feature_frame.frame_index
                            ),
                            file=sys.stderr,
                        )
                    self.last_stats_time = now

                if self.args.max_frames is not None and feature_frame.frame_index + 1 >= self.args.max_frames:
                    break
                if sleep_seconds:
                    time.sleep(sleep_seconds)


class nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, traceback):
        return False


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Run AvatarPoser from NOLO/head-hand tracking and print full-body joints."
    )
    parser.add_argument("--opt", default="options/test_avatarposer.json", help="AvatarPoser option json.")
    parser.add_argument("--checkpoint", default=None, help="Override pretrained checkpoint path.")
    parser.add_argument(
        "--root-orientation-mode",
        choices=("head_yaw", "model", "head"),
        default="head_yaw",
        help=(
            "head_yaw uses only HMD yaw for body turning; model uses AvatarPoser's predicted root orientation; "
            "head derives full root orientation from tracked head pose."
        ),
    )
    parser.add_argument(
        "--head-yaw-axis",
        choices=("x", "y", "z", "-x", "-y", "-z"),
        default="x",
        help="Head local axis projected to the horizontal plane for HMD-driven body yaw in head_yaw mode.",
    )
    parser.add_argument(
        "--head-yaw-offset-deg",
        type=float,
        default=0.0,
        help="Additional yaw offset in degrees for HMD-driven body turning.",
    )
    parser.add_argument(
        "--no-upright-constraint",
        action="store_false",
        dest="upright_constraint",
        help="Disable realtime root-orientation upright constraint and allow the raw model root orientation.",
    )
    parser.set_defaults(upright_constraint=True)
    parser.add_argument(
        "--source",
        default="auto",
        help="Input source: auto, stdin, udp, or a data_fps60 .pkl file.",
    )
    parser.add_argument("--udp-host", default="127.0.0.1", help="UDP host for NOLO tracking exporter.")
    parser.add_argument("--udp-port", type=int, default=39001, help="UDP port for NOLO tracking exporter.")
    parser.add_argument(
        "--no-drop-stale-udp",
        action="store_true",
        help="Process every UDP packet. Default drops stale packets to reduce realtime latency.",
    )
    parser.add_argument(
        "--udp-receive-buffer",
        type=int,
        default=65536,
        help="UDP receive buffer size in bytes. Smaller buffers reduce latency when rendering is slow.",
    )
    parser.add_argument(
        "--tracking-coordinates",
        choices=coordinate_transform_names(),
        default="htc_mr",
        help="Coordinate system for live JSON/UDP 6DoF tracking. Default assumes HTC-MR/AvatarPoser coordinates.",
    )
    parser.add_argument(
        "--orientation-transform",
        choices=("pose", "basis"),
        default="pose",
        help="Rotation conversion mode used only when a non-identity coordinate transform is selected.",
    )
    parser.add_argument(
        "--wrist-offset-preset",
        choices=wrist_offset_preset_names(),
        default="none",
        help="Optional controller-to-wrist rotation preset. Explicit left/right offsets override the preset.",
    )
    parser.add_argument(
        "--tracking-calibration",
        default="options/realtime_calibration.json",
        help="JSON local transforms from tracked NOLO/HMD/controller poses to AvatarPoser head/wrist joint poses.",
    )
    parser.add_argument(
        "--swap-hands",
        action="store_true",
        help="Swap incoming left/right controller poses before building AvatarPoser left_wrist/right_wrist features.",
    )
    parser.add_argument(
        "--height-calibration-mode",
        choices=("lock", "copy"),
        default="lock",
        help=(
            "A-key height calibration mode. lock stores the current model-axis height as standing height; "
            "copy keeps copying source-axis height into the model height axis every frame after A is pressed."
        ),
    )
    parser.add_argument(
        "--height-source-axis",
        choices=("x", "y", "z"),
        default="z",
        help=(
            "Input feature axis copied into the model height axis in copy mode. "
            "Lock mode records the current model height axis instead."
        ),
    )
    parser.add_argument(
        "--height-model-axis",
        choices=("x", "y", "z"),
        default="z",
        help="Model feature axis used as AvatarPoser's vertical height. AvatarPoser AMASS data is Z-up.",
    )
    parser.add_argument(
        "--standing-head-height",
        type=float,
        default=1.5,
        help=(
            "Target model head height in meters when A is pressed in lock mode. "
            "Use 0 to lock the currently observed model height instead."
        ),
    )
    parser.add_argument(
        "--left-wrist-offset-deg",
        type=parse_degrees,
        default=None,
        help="Optional local XYZ Euler offset in degrees applied after the converted left controller rotation.",
    )
    parser.add_argument(
        "--right-wrist-offset-deg",
        type=parse_degrees,
        default=None,
        help="Optional local XYZ Euler offset in degrees applied after the converted right controller rotation.",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames.")
    parser.add_argument("--fps", type=float, default=0.0, help="Replay rate. 0 means no sleep.")
    parser.add_argument("--print-every", type=int, default=1, help="Print joints every N frames. 0 disables printing.")
    parser.add_argument(
        "--debug-tracking-every",
        type=int,
        default=0,
        help="Print converted head/hand tracking diagnostics every N input frames. 0 disables diagnostics.",
    )
    parser.add_argument(
        "--debug-rotation-every",
        type=int,
        default=0,
        help="Print converted head/hand local rotation axes every N input frames. 0 disables diagnostics.",
    )
    parser.add_argument(
        "--debug-root-every",
        type=int,
        default=0,
        help="Print predicted root orientation and key joint heights every N output frames. 0 disables diagnostics.",
    )
    parser.add_argument("--export-jsonl", default=None, help="Optional path to save joint payloads.")
    parser.add_argument("--visualize", action="store_true", help="Show a realtime visualization window.")
    parser.add_argument(
        "--visualizer",
        choices=("auto", "body", "mesh", "joints"),
        default="auto",
        help="Visualization backend. auto tries body_visualizer, then built-in mesh, then built-in joints.",
    )
    parser.add_argument(
        "--visualize-fps",
        type=float,
        default=30.0,
        help="Maximum visualization refresh rate. 0 renders every processed frame.",
    )
    parser.add_argument(
        "--mesh-faces",
        type=int,
        default=2500,
        help="Maximum triangle faces rendered by the built-in mesh visualizer.",
    )
    parser.add_argument(
        "--visual-smoothing",
        type=float,
        default=0.15,
        help="Display-only exponential smoothing for built-in visualizers. 0 disables smoothing.",
    )
    parser.add_argument(
        "--visual-camera-mode",
        choices=("world", "follow"),
        default="follow",
        help="follow keeps the avatar centered in the window; world keeps the first pelvis as a fixed camera anchor.",
    )
    parser.add_argument(
        "--stats-every",
        type=int,
        default=0,
        help="Print processing statistics every N processed frames. 0 disables stats.",
    )
    parser.add_argument("--width", type=int, default=800, help="Visualizer width.")
    parser.add_argument("--height", type=int, default=800, help="Visualizer height.")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    if args.export_jsonl:
        export_dir = os.path.dirname(args.export_jsonl)
        if export_dir:
            os.makedirs(export_dir, exist_ok=True)
    RealtimeAvatarPoserApp(args).run()


if __name__ == "__main__":
    main(sys.argv[1:])
