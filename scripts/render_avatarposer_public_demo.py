import argparse
import os
import sys
from collections import deque

import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from avatarposer_realtime.runtime_env import configure_runtime_environment

configure_runtime_environment()

from avatarposer_realtime.model_runtime import AvatarPoserRuntime
from avatarposer_realtime.sources import PklReplaySource, default_pkl_source
from avatarposer_realtime.tk_visualizer import SKELETON_EDGES, TkCameraProjector, copy_to_numpy
from avatarposer_realtime.types import JOINT_NAMES


class OffscreenMeshRenderer:
    def __init__(self, body_model, width=800, height=800, max_faces=4500, view_scale=0.52):
        self.width = int(width)
        self.height = int(height)
        self.projector = TkCameraProjector(width, height)
        self.projector.scale = min(self.width, self.height) * float(view_scale)
        self.joint_indices = {name: index for index, name in enumerate(JOINT_NAMES)}
        self.faces = copy_to_numpy(body_model.f, dtype=np.int32)
        self.render_faces = self.select_faces(self.faces, int(max_faces))

    @staticmethod
    def select_faces(faces, max_faces):
        if max_faces <= 0 or len(faces) <= max_faces:
            return faces
        indices = np.linspace(0, len(faces) - 1, max_faces, dtype=np.int32)
        return faces[indices]

    def draw_floor(self, draw):
        horizon = int(self.height * 0.66)
        draw.rectangle([0, horizon, self.width, self.height], fill=(207, 217, 215))
        for i in range(18):
            t = i / 17.0
            y = int(horizon + (self.height - horizon) * (t * t))
            color = (169, 183, 181) if i % 2 == 0 else (231, 236, 235)
            draw.line([(0, y), (self.width, y)], fill=color, width=1)
        for i in range(-12, 13):
            x = self.width * 0.5 + i * self.width * 0.045
            draw.line([(x, self.height), (self.width * 0.5 + i * 5, horizon)], fill=(169, 183, 181), width=1)

    def face_colors(self, vertices, faces):
        triangles = vertices[faces]
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        normal_norm = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.maximum(normal_norm, 1e-6)
        light = np.asarray([0.25, 0.85, -0.45], dtype=np.float32)
        light = light / np.linalg.norm(light)
        diffuse = np.clip(np.matmul(normals, light), 0.0, 1.0)
        shade = 0.68 + diffuse * 0.32
        base = np.asarray([213, 229, 177], dtype=np.float32)
        return np.clip(base[None, :] * shade[:, None], 0, 255).astype(np.uint8)

    def draw_mesh(self, draw, vertices, root):
        projected = self.projector.project(vertices, root)
        faces = self.render_faces
        face_depth = projected[faces, 2].mean(axis=1)
        colors = self.face_colors(vertices, faces)
        for face_index in np.argsort(face_depth):
            face = faces[face_index]
            points = [(float(projected[index, 0]), float(projected[index, 1])) for index in face]
            draw.polygon(points, fill=tuple(int(value) for value in colors[face_index]))

    def draw_joint_overlay(self, draw, joints):
        root = joints[self.joint_indices["pelvis"]]
        points = {name: self.projector.project_one(joints[index], root) for name, index in self.joint_indices.items()}
        for start, end in SKELETON_EDGES:
            draw.line([points[start], points[end]], fill=(98, 111, 105), width=2)
        for name in ("head", "left_wrist", "right_wrist"):
            x, y = points[name]
            radius = 4
            draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=(99, 120, 74))

    def render(self, result):
        vertices = copy_to_numpy(result.body.v[0])
        joints = copy_to_numpy(result.joints)
        root = joints[self.joint_indices["pelvis"]]

        image = Image.new("RGB", (self.width, self.height), (246, 247, 244))
        draw = ImageDraw.Draw(image)
        text = "full-body pose of the user's avatar"
        text_x = int(self.width * 0.5 - 150)
        draw.text((text_x, 22), text, fill=(125, 129, 127))
        self.draw_floor(draw)
        self.draw_mesh(draw, vertices, root)
        self.draw_joint_overlay(draw, joints)
        return image


class SparseInputRenderer:
    def __init__(self, width=800, height=800, view_scale=0.52):
        self.width = int(width)
        self.height = int(height)
        self.projector = TkCameraProjector(width, height, up_axis="y")
        self.projector.scale = min(self.width, self.height) * float(view_scale)

    def render(self, feature_frame):
        positions = np.asarray(feature_frame.feature[36:45], dtype=np.float32).reshape(3, 3)
        root = positions[0]
        projected = self.projector.project(positions, root)
        image = Image.new("RGB", (self.width, self.height), (245, 245, 242))
        draw = ImageDraw.Draw(image)
        draw.text((24, 22), "6D pose of Mixed Reality headset and controllers", fill=(125, 129, 127))

        horizon = int(self.height * 0.66)
        draw.rectangle([0, horizon, self.width, self.height], fill=(224, 229, 226))
        for i in range(16):
            y = int(horizon + (self.height - horizon) * ((i / 15.0) ** 2))
            draw.line([(0, y), (self.width, y)], fill=(180, 190, 188), width=1)

        labels = ("HMD", "Left", "Right")
        colors = ((45, 65, 72), (42, 126, 156), (182, 77, 61))
        for index, label in enumerate(labels):
            x, y = float(projected[index, 0]), float(projected[index, 1])
            radius = 12 if index == 0 else 10
            color = colors[index]
            draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color, outline=(255, 255, 255), width=2)
            draw.text((x + 14, y - 8), label, fill=color)
        draw.line([(float(projected[0, 0]), float(projected[0, 1])), (float(projected[1, 0]), float(projected[1, 1]))], fill=(128, 137, 132), width=2)
        draw.line([(float(projected[0, 0]), float(projected[0, 1])), (float(projected[2, 0]), float(projected[2, 1]))], fill=(128, 137, 132), width=2)
        return image


def combine_side_by_side(left_image, right_image):
    width = left_image.width + right_image.width
    height = max(left_image.height, right_image.height)
    canvas = Image.new("RGB", (width, height), (12, 16, 20))
    canvas.paste(left_image, (0, 0))
    canvas.paste(right_image, (left_image.width, 0))
    return canvas


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Render the public AvatarPoser pretrained demo from AMASS sparse MR inputs.")
    parser.add_argument("--source", default="auto", help="Preprocessed data_fps60 .pkl file, or auto for the first test clip.")
    parser.add_argument("--output-dir", default="results/public_demo", help="Directory for PNG/GIF outputs.")
    parser.add_argument("--frames", type=int, default=120, help="Number of rendered frames.")
    parser.add_argument("--stride", type=int, default=1, help="Render every Nth source frame.")
    parser.add_argument("--fps", type=int, default=20, help="GIF playback FPS.")
    parser.add_argument("--width", type=int, default=640, help="Single panel width.")
    parser.add_argument("--height", type=int, default=640, help="Single panel height.")
    parser.add_argument("--view-scale", type=float, default=0.52, help="Camera scale relative to panel size.")
    parser.add_argument("--mesh-faces", type=int, default=4500, help="Max mesh faces. 0 renders all faces.")
    parser.add_argument("--checkpoint", default=None, help="Override model_zoo/avatarposer.pth.")
    parser.add_argument("--side-by-side", action="store_true", help="Render sparse MR input panel next to avatar panel.")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)

    source_path = default_pkl_source() if args.source == "auto" else args.source
    runtime = AvatarPoserRuntime(checkpoint=args.checkpoint, root_orientation_mode="model")
    source = PklReplaySource(source_path)
    mesh_renderer = OffscreenMeshRenderer(runtime.body_model, args.width, args.height, args.mesh_faces, args.view_scale)
    sparse_renderer = SparseInputRenderer(args.width, args.height, args.view_scale) if args.side_by_side else None

    history = deque(maxlen=runtime.window_size)
    frames = []
    rendered = 0
    for source_index, feature_frame in enumerate(source):
        history.append(feature_frame.feature)
        if source_index % max(args.stride, 1) != 0:
            continue
        result = runtime.predict(feature_frame.frame_index, np.stack(history, axis=0), feature_frame.head_transform)
        avatar_image = mesh_renderer.render(result)
        if sparse_renderer is not None:
            sparse_image = sparse_renderer.render(feature_frame)
            avatar_image = combine_side_by_side(sparse_image, avatar_image)
        frames.append(avatar_image)
        rendered += 1
        if rendered >= args.frames:
            break

    if not frames:
        raise RuntimeError("No frames were rendered from source: {}".format(source_path))

    first_png = os.path.join(args.output_dir, "avatarposer_public_demo_first.png")
    middle_png = os.path.join(args.output_dir, "avatarposer_public_demo_middle.png")
    gif_path = os.path.join(args.output_dir, "avatarposer_public_demo.gif")

    frames[0].save(first_png)
    frames[len(frames) // 2].save(middle_png)
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=max(int(1000 / max(args.fps, 1)), 1),
        loop=0,
        optimize=False,
    )

    print("source={}".format(source_path))
    print("frames={}".format(len(frames)))
    print("first_png={}".format(os.path.abspath(first_png)))
    print("middle_png={}".format(os.path.abspath(middle_png)))
    print("gif={}".format(os.path.abspath(gif_path)))


if __name__ == "__main__":
    main(sys.argv[1:])
