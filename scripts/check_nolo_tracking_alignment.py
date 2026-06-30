import argparse
import os
import sys


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from avatarposer_realtime.runtime_env import configure_runtime_environment

configure_runtime_environment()

from avatarposer_realtime.calibration import wrist_offset_preset_names
from avatarposer_realtime.diagnostics import RotationAxesDebugger, TrackingDebugger
from avatarposer_realtime.sources import create_source
from avatarposer_realtime.transforms import coordinate_transform_names


def parse_degrees(value):
    values = [float(item.strip()) for item in value.split(",")]
    if len(values) != 3:
        raise argparse.ArgumentTypeError("Expected three comma-separated degree values, e.g. 0,0,90")
    return values


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Inspect NOLO/OpenVR tracking coordinate and hand rotation axes.")
    parser.add_argument("--udp-host", default="127.0.0.1")
    parser.add_argument("--udp-port", type=int, default=39001)
    parser.add_argument(
        "--tracking-coordinates",
        choices=coordinate_transform_names(),
        default="htc_mr",
    )
    parser.add_argument(
        "--orientation-transform",
        choices=("pose", "basis"),
        default="pose",
    )
    parser.add_argument(
        "--wrist-offset-preset",
        choices=wrist_offset_preset_names(),
        default="none",
    )
    parser.add_argument(
        "--tracking-calibration",
        default="options/realtime_calibration.json",
    )
    parser.add_argument(
        "--swap-hands",
        action="store_true",
    )
    parser.add_argument("--left-wrist-offset-deg", type=parse_degrees, default=None)
    parser.add_argument("--right-wrist-offset-deg", type=parse_degrees, default=None)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--every", type=int, default=30)
    parser.add_argument("--no-drop-stale-udp", action="store_true")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    source = create_source(
        "udp",
        udp_host=args.udp_host,
        udp_port=args.udp_port,
        latest_only=not args.no_drop_stale_udp,
        coordinate_system=args.tracking_coordinates,
        orientation_transform=args.orientation_transform,
        wrist_offset_preset=args.wrist_offset_preset,
        left_wrist_offset_degrees=args.left_wrist_offset_deg,
        right_wrist_offset_degrees=args.right_wrist_offset_deg,
        tracking_calibration_path=args.tracking_calibration,
        swap_hands=args.swap_hands,
    )
    tracking_debugger = TrackingDebugger(every=args.every)
    rotation_debugger = RotationAxesDebugger(every=args.every)
    for index, feature_frame in enumerate(source):
        tracking_debugger.update(feature_frame)
        rotation_debugger.update(feature_frame)
        if index + 1 >= args.frames:
            break


if __name__ == "__main__":
    main(sys.argv[1:])
