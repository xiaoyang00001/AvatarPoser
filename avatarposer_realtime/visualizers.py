import json
import sys

import numpy as np

from avatarposer_realtime.tk_visualizer import TkJointVisualizer, TkMeshVisualizer
from avatarposer_realtime.types import JOINT_NAMES


def joints_payload(result, decimals=6):
    return {
        "frame": int(result.frame_index),
        "joints": {
            name: [round(float(value), decimals) for value in result.joints[joint_index]]
            for joint_index, name in enumerate(JOINT_NAMES)
        },
    }


class JointPrinter:
    def __init__(self, every=1, stream=None):
        self.every = int(every)
        self.stream = stream or sys.stdout

    def update(self, result):
        if self.every <= 0 or result.frame_index % self.every != 0:
            return None
        text = json.dumps(joints_payload(result), ensure_ascii=False)
        print(text, file=self.stream)
        return text


class JsonlExporter:
    def __init__(self, path):
        self.path = path
        self.file = None

    def __enter__(self):
        self.file = open(self.path, "w", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self.file:
            self.file.close()
        return False

    def update(self, result):
        self.file.write(json.dumps(joints_payload(result), ensure_ascii=False) + "\n")
        self.file.flush()


class BodyVisualizerRenderer:
    def __init__(self, body_model, width=800, height=800):
        from body_visualizer.mesh.mesh_viewer import MeshViewer
        from body_visualizer.tools.vis_tools import colors
        from human_body_prior.tools.omni_tools import copy2cpu
        import trimesh

        self.copy2cpu = copy2cpu
        self.colors = colors
        self.trimesh = trimesh
        self.viewer = MeshViewer(width=width, height=height, use_offscreen=False)
        self.faces = self.copy2cpu(body_model.f)

    def update(self, result):
        vertices = self.copy2cpu(result.body.v[0])
        body_mesh = self.trimesh.Trimesh(
            vertices=vertices,
            faces=self.faces,
            vertex_colors=np.tile(self.colors["purple"], (vertices.shape[0], 1)),
            process=False,
        )
        self.viewer.set_static_meshes([body_mesh])


def create_body_visualizer(
    body_model,
    width=800,
    height=800,
    backend="auto",
    mesh_faces=2500,
    smoothing=0.15,
    camera_mode="follow",
):
    if backend == "joints":
        return TkJointVisualizer(width=width, height=height, smoothing=smoothing, camera_mode=camera_mode)
    if backend == "mesh":
        return TkMeshVisualizer(
            body_model,
            width=width,
            height=height,
            max_faces=mesh_faces,
            smoothing=smoothing,
            camera_mode=camera_mode,
        )

    try:
        return BodyVisualizerRenderer(body_model, width=width, height=height)
    except Exception as exc:
        print("body_visualizer is not available: {}".format(exc), file=sys.stderr)
        if backend == "body":
            print("Continuing with joint output only.", file=sys.stderr)
            return None
        print("Falling back to built-in mesh visualizer.", file=sys.stderr)
        try:
            return TkMeshVisualizer(
                body_model,
                width=width,
                height=height,
                max_faces=mesh_faces,
                smoothing=smoothing,
                camera_mode=camera_mode,
            )
        except Exception as mesh_exc:
            print("Built-in mesh visualizer is not available: {}".format(mesh_exc), file=sys.stderr)
            print("Falling back to built-in joint visualizer.", file=sys.stderr)
        try:
            return TkJointVisualizer(width=width, height=height, smoothing=smoothing, camera_mode=camera_mode)
        except Exception as fallback_exc:
            print("Built-in joint visualizer is not available: {}".format(fallback_exc), file=sys.stderr)
            print("Continuing with joint output only.", file=sys.stderr)
        return None
