import numpy as np

from avatarposer_realtime.types import JOINT_NAMES


SKELETON_EDGES = [
    ("pelvis", "spine1"),
    ("spine1", "spine2"),
    ("spine2", "spine3"),
    ("spine3", "neck"),
    ("neck", "head"),
    ("spine3", "left_collar"),
    ("left_collar", "left_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("spine3", "right_collar"),
    ("right_collar", "right_shoulder"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("pelvis", "left_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("left_ankle", "left_foot"),
    ("pelvis", "right_hip"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("right_ankle", "right_foot"),
]


def copy_to_numpy(value, dtype=np.float32):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


class TkVisualizerBase:
    def __init__(self, title, width=800, height=800):
        import tkinter as tk

        self.tk = tk
        self.width = int(width)
        self.height = int(height)
        self.closed = False
        self.height_calibration_requests = 0
        self.root = tk.Tk()
        self.root.title(title)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind_all("<KeyPress>", self.on_key_press)
        self.canvas = tk.Canvas(self.root, width=self.width, height=self.height, bg="#f6f7f4", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.focus_set()
        self.root.update()

    def close(self):
        self.closed = True
        self.root.destroy()

    def on_key_press(self, event):
        if getattr(event, "keysym", "").lower() == "a":
            self.height_calibration_requests += 1

    def consume_height_calibration_request(self):
        if self.height_calibration_requests <= 0:
            return False
        self.height_calibration_requests -= 1
        return True

    def poll_events(self):
        if not self.closed:
            self.update_window()

    def update_window(self):
        self.root.update_idletasks()
        self.root.update()


class TkCameraProjector:
    def __init__(self, width, height, up_axis="z"):
        self.width = int(width)
        self.height = int(height)
        self.scale = min(self.width, self.height) * 0.72
        self.up_axis = up_axis

    def project(self, points, root):
        points = copy_to_numpy(points)
        root = copy_to_numpy(root).reshape(3)
        centered = points - root
        if self.up_axis == "y":
            x = self.width * 0.5 + centered[:, 0] * self.scale - centered[:, 2] * self.scale * 0.18
            y = self.height * 0.78 - centered[:, 1] * self.scale + centered[:, 2] * self.scale * 0.10
            depth = centered[:, 2] - centered[:, 0] * 0.06
        else:
            x = self.width * 0.5 - centered[:, 1] * self.scale + centered[:, 0] * self.scale * 0.18
            y = self.height * 0.78 - centered[:, 2] * self.scale + centered[:, 0] * self.scale * 0.10
            depth = centered[:, 0] - centered[:, 1] * 0.06
        return np.stack([x, y, depth], axis=1)

    def project_one(self, point, root):
        projected = self.project(np.asarray(point, dtype=np.float32).reshape(1, 3), root)
        return float(projected[0, 0]), float(projected[0, 1])


class TkJointVisualizer(TkVisualizerBase):
    def __init__(self, width=800, height=800, smoothing=0.15, camera_mode="follow"):
        super().__init__("AvatarPoser Realtime Joints", width=width, height=height)
        self.joint_indices = {name: index for index, name in enumerate(JOINT_NAMES)}
        self.projector = TkCameraProjector(width, height)
        self.smoothing = float(np.clip(smoothing, 0.0, 0.95))
        self.camera_mode = camera_mode
        self.camera_root = None
        self.previous_joints = None

    def camera_anchor(self, joints):
        pelvis = joints[self.joint_indices["pelvis"]]
        if self.camera_mode == "follow":
            return pelvis
        if self.camera_root is None:
            self.camera_root = pelvis.copy()
        return self.camera_root

    def draw_floor(self):
        horizon = self.height * 0.66
        bottom = self.height
        self.canvas.create_rectangle(0, horizon, self.width, bottom, fill="#cfd9d7", outline="")
        for i in range(16):
            t = i / 15.0
            y = horizon + (bottom - horizon) * (t * t)
            color = "#a9b7b5" if i % 2 == 0 else "#e7eceb"
            self.canvas.create_line(0, y, self.width, y, fill=color)
        for i in range(-8, 9):
            x = self.width * 0.5 + i * self.width * 0.055
            self.canvas.create_line(x, bottom, self.width * 0.5 + i * 6, horizon, fill="#a9b7b5")

    def update(self, result):
        if self.closed:
            return
        joints = copy_to_numpy(result.joints)
        if self.previous_joints is not None and self.smoothing > 0.0:
            joints = self.previous_joints * self.smoothing + joints * (1.0 - self.smoothing)
        self.previous_joints = joints.copy()
        root = self.camera_anchor(joints)
        points = {name: self.projector.project_one(joints[index], root) for name, index in self.joint_indices.items()}

        self.canvas.delete("all")
        self.canvas.create_text(
            self.width * 0.5,
            32,
            text="full-body pose of the user's avatar",
            fill="#7d817f",
            font=("Segoe UI", 16),
        )
        self.draw_floor()

        for start, end in SKELETON_EDGES:
            x1, y1 = points[start]
            x2, y2 = points[end]
            self.canvas.create_line(x1, y1, x2, y2, fill="#65726d", width=5, capstyle=self.tk.ROUND)
            self.canvas.create_line(x1, y1, x2, y2, fill="#dce8be", width=3, capstyle=self.tk.ROUND)

        for name, (x, y) in points.items():
            radius = 6
            fill = "#b9cf85"
            if name in ("head", "left_wrist", "right_wrist"):
                fill = "#90a95f"
                radius = 8
            self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=fill, outline="#65726d")

        self.update_window()


class TkMeshVisualizer(TkVisualizerBase):
    def __init__(self, body_model, width=800, height=800, max_faces=4500, smoothing=0.15, camera_mode="follow"):
        super().__init__("AvatarPoser Realtime Mesh", width=width, height=height)
        from PIL import Image, ImageDraw, ImageTk

        self.image_module = Image
        self.draw_module = ImageDraw
        self.image_tk_module = ImageTk
        self.projector = TkCameraProjector(width, height)
        self.joint_indices = {name: index for index, name in enumerate(JOINT_NAMES)}
        self.faces = copy_to_numpy(body_model.f, dtype=np.int32)
        self.render_faces = self.select_faces(self.faces, max_faces)
        self.photo_image = None
        self.image_item = None
        self.smoothing = float(np.clip(smoothing, 0.0, 0.95))
        self.camera_mode = camera_mode
        self.camera_root = None
        self.previous_vertices = None
        self.previous_joints = None

    def camera_anchor(self, joints):
        pelvis = joints[self.joint_indices["pelvis"]]
        if self.camera_mode == "follow":
            return pelvis
        if self.camera_root is None:
            self.camera_root = pelvis.copy()
        return self.camera_root

    @staticmethod
    def select_faces(faces, max_faces):
        if max_faces <= 0 or len(faces) <= max_faces:
            return faces
        indices = np.linspace(0, len(faces) - 1, int(max_faces), dtype=np.int32)
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
            color = tuple(int(value) for value in colors[face_index])
            draw.polygon(points, fill=color)

    def draw_joint_overlay(self, draw, joints):
        root = self.camera_anchor(joints)
        points = {name: self.projector.project_one(joints[index], root) for name, index in self.joint_indices.items()}
        for start, end in SKELETON_EDGES:
            draw.line([points[start], points[end]], fill=(98, 111, 105), width=2)
        for name in ("head", "left_wrist", "right_wrist"):
            x, y = points[name]
            r = 4
            draw.ellipse([x - r, y - r, x + r, y + r], fill=(99, 120, 74))

    def update(self, result):
        if self.closed:
            return
        vertices = copy_to_numpy(result.body.v[0])
        joints = copy_to_numpy(result.joints)
        if self.previous_vertices is not None and self.smoothing > 0.0:
            vertices = self.previous_vertices * self.smoothing + vertices * (1.0 - self.smoothing)
            joints = self.previous_joints * self.smoothing + joints * (1.0 - self.smoothing)
        self.previous_vertices = vertices.copy()
        self.previous_joints = joints.copy()
        root = self.camera_anchor(joints)

        image = self.image_module.new("RGB", (self.width, self.height), (246, 247, 244))
        draw = self.draw_module.Draw(image)
        draw.text((self.width * 0.5 - 150, 22), "full-body pose of the user's avatar", fill=(125, 129, 127))
        self.draw_floor(draw)
        self.draw_mesh(draw, vertices, root)
        self.draw_joint_overlay(draw, joints)

        self.photo_image = self.image_tk_module.PhotoImage(image=image)
        if self.image_item is None:
            self.image_item = self.canvas.create_image(0, 0, anchor=self.tk.NW, image=self.photo_image)
        else:
            self.canvas.itemconfigure(self.image_item, image=self.photo_image)
        self.update_window()
