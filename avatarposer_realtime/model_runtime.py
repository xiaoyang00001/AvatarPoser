import torch

from models.select_model import define_Model
from utils import utils_option, utils_transform
from utils.rotation_tools import aa2matrot, local2global_pose, matrot2aa

from avatarposer_realtime.types import AvatarPoseResult


class AvatarPoserRuntime:
    def __init__(
        self,
        opt_path="options/test_avatarposer.json",
        checkpoint=None,
        root_orientation_mode="model",
        upright_constraint=False,
        head_yaw_axis="x",
        head_yaw_offset_degrees=0.0,
    ):
        self.opt = utils_option.parse(opt_path, is_train=True)
        if checkpoint:
            self.opt["path"]["pretrained_netG"] = checkpoint
            self.opt["path"]["pretrained"] = checkpoint
        self.opt = utils_option.dict_to_nonedict(self.opt)

        self.model = define_Model(self.opt)
        self.model.init_test()
        self.model.netG.eval()

        self.device = self.model.device
        self.net = self.model.netG.module
        self.body_model = self.model.bm
        self.window_size = self.opt["datasets"]["test"]["window_size"]
        self.root_orientation_mode = root_orientation_mode
        self.upright_constraint = bool(upright_constraint)
        self.head_yaw_axis = head_yaw_axis
        self.head_yaw_offset = torch.deg2rad(
            torch.tensor(float(head_yaw_offset_degrees), device=self.device, dtype=torch.float32)
        )

    @staticmethod
    def parse_signed_axis(axis_name):
        sign = -1.0 if axis_name.startswith("-") else 1.0
        axis = axis_name[1:] if axis_name.startswith("-") else axis_name
        axis_indices = {"x": 0, "y": 1, "z": 2}
        if axis not in axis_indices:
            raise ValueError("Unknown head yaw axis: {}".format(axis_name))
        return axis_indices[axis], sign

    def constrain_root_upright(self, root_orientation):
        root_rotation = aa2matrot(root_orientation.reshape(-1, 3)).reshape(-1, 3, 3)
        yaw = torch.atan2(root_rotation[:, 1, 0], root_rotation[:, 0, 0])
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        zeros = torch.zeros_like(yaw)
        ones = torch.ones_like(yaw)
        upright_rotation = torch.stack(
            [
                torch.stack([cos_yaw, -sin_yaw, zeros], dim=1),
                torch.stack([sin_yaw, cos_yaw, zeros], dim=1),
                torch.stack([zeros, zeros, ones], dim=1),
            ],
            dim=1,
        )
        return matrot2aa(upright_rotation).to(dtype=root_orientation.dtype)

    def rotation_matrix_from_axis_angle(self, axis, angle):
        axis = axis / torch.clamp(torch.linalg.norm(axis), min=1e-6)
        x, y, z = axis
        zero = torch.zeros((), device=axis.device, dtype=axis.dtype)
        skew = torch.stack(
            [
                torch.stack([zero, -z, y]),
                torch.stack([z, zero, -x]),
                torch.stack([-y, x, zero]),
            ]
        )
        identity = torch.eye(3, device=axis.device, dtype=axis.dtype)
        return identity + torch.sin(angle) * skew + (1.0 - torch.cos(angle)) * torch.matmul(skew, skew)

    def rotation_to_upright(self, body_up):
        target = torch.tensor([0.0, 0.0, 1.0], device=body_up.device, dtype=body_up.dtype)
        rotations = []
        for vector in body_up:
            norm = torch.linalg.norm(vector)
            if norm < 1e-6:
                rotations.append(torch.eye(3, device=body_up.device, dtype=body_up.dtype))
                continue

            source = vector / norm
            cross = torch.linalg.cross(source, target, dim=0)
            cross_norm = torch.linalg.norm(cross)
            dot = torch.clamp(torch.dot(source, target), -1.0, 1.0)
            if cross_norm < 1e-6:
                if dot > 0.0:
                    rotations.append(torch.eye(3, device=body_up.device, dtype=body_up.dtype))
                    continue
                fallback = torch.tensor([1.0, 0.0, 0.0], device=body_up.device, dtype=body_up.dtype)
                if torch.abs(torch.dot(source, fallback)) > 0.9:
                    fallback = torch.tensor([0.0, 1.0, 0.0], device=body_up.device, dtype=body_up.dtype)
                axis = torch.linalg.cross(source, fallback, dim=0)
                rotations.append(self.rotation_matrix_from_axis_angle(axis, torch.as_tensor(torch.pi, device=body_up.device, dtype=body_up.dtype)))
                continue

            axis = cross / cross_norm
            angle = torch.atan2(cross_norm, dot)
            rotations.append(self.rotation_matrix_from_axis_angle(axis, angle))

        return torch.stack(rotations, dim=0)

    def constrain_body_output_upright(self, body, head_transform):
        joints = body.Jtr
        pelvis = joints[:, 0, :]
        head = joints[:, 15, :]
        body_up = head - pelvis
        rotation = self.rotation_to_upright(body_up)
        source_anchor = head
        target_anchor = head_transform[:, :3, 3].to(device=joints.device, dtype=joints.dtype)

        def rotate_points(points):
            centered = points - source_anchor[:, None, :]
            rotated = torch.matmul(centered, rotation.transpose(1, 2))
            return rotated + target_anchor[:, None, :]

        corrected_joints = rotate_points(body.Jtr)
        corrected_vertices = rotate_points(body.v)
        body.Jtr = corrected_joints
        body.v = corrected_vertices
        body.vertices = corrected_vertices
        return body, corrected_joints[:, :22, :]

    def root_translation_from_head_position(self, predicted_angle, root_orientation, head_transform):
        body_pose_local = self.body_model(
            pose_body=predicted_angle[..., 3:66],
            root_orient=root_orientation,
        )
        head_to_root = body_pose_local.Jtr[:, 15, :]
        return -head_to_root + head_transform[:, :3, 3].to(dtype=predicted_angle.dtype)

    def root_orientation_from_head_yaw(self, head_transform, dtype):
        axis_index, axis_sign = self.parse_signed_axis(self.head_yaw_axis)
        head_rotation = head_transform[:, :3, :3]
        heading = head_rotation[:, :2, axis_index] * axis_sign
        heading_norm = torch.linalg.norm(heading, dim=1)
        fallback = torch.tensor([1.0, 0.0], device=head_transform.device, dtype=head_transform.dtype).repeat(
            head_transform.shape[0], 1
        )
        heading = torch.where((heading_norm > 1e-5).unsqueeze(1), heading, fallback)
        yaw = torch.atan2(heading[:, 1], heading[:, 0]) + self.head_yaw_offset.to(head_transform.device)
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        zeros = torch.zeros_like(yaw)
        ones = torch.ones_like(yaw)
        rotation = torch.stack(
            [
                torch.stack([cos_yaw, -sin_yaw, zeros], dim=1),
                torch.stack([sin_yaw, cos_yaw, zeros], dim=1),
                torch.stack([zeros, zeros, ones], dim=1),
            ],
            dim=1,
        )
        return matrot2aa(rotation).to(dtype=dtype)

    def root_transform_from_head(self, predicted_angle, head_transform):
        batch_size = predicted_angle.shape[0]
        zeros_root = torch.zeros(batch_size, 3, device=self.device, dtype=predicted_angle.dtype)
        local_angle = torch.cat([zeros_root, predicted_angle[..., 3:66]], dim=1)
        local_rotation = aa2matrot(local_angle.reshape(-1, 3)).reshape(batch_size, -1, 9)
        global_rotation = local2global_pose(local_rotation, self.body_model.kintree_table[0][:22].long())
        root_to_head_rotation = global_rotation[:, 15, :, :]

        body_pose_local = self.body_model(pose_body=predicted_angle[..., 3:66])
        root_to_head_translation = body_pose_local.Jtr[:, 15, :]

        root_to_head = torch.eye(4, device=self.device, dtype=head_transform.dtype).repeat(batch_size, 1, 1)
        root_to_head[:, :3, :3] = root_to_head_rotation.to(dtype=head_transform.dtype)
        root_to_head[:, :3, 3] = root_to_head_translation.to(dtype=head_transform.dtype)

        root_to_world = torch.matmul(head_transform, torch.inverse(root_to_head))
        root_orientation = matrot2aa(root_to_world[:, :3, :3]).to(dtype=predicted_angle.dtype)
        root_translation = root_to_world[:, :3, 3].to(dtype=predicted_angle.dtype)
        return root_orientation, root_translation

    def predict(self, frame_index, feature_window, head_transform):
        features = torch.as_tensor(feature_window, dtype=torch.float32, device=self.device)
        if features.ndim == 1:
            features = features.unsqueeze(0)
        features = features[-self.window_size :]
        head_transform = torch.as_tensor(head_transform, dtype=torch.float32, device=self.device).reshape(1, 4, 4)

        with torch.no_grad():
            global_orientation_6d, joint_rotation_6d = self.net(features.unsqueeze(0), do_fk=False)
            prediction_6d = torch.cat([global_orientation_6d, joint_rotation_6d], dim=-1)
            predicted_angle = utils_transform.sixd2aa(prediction_6d.reshape(-1, 6)).reshape(1, -1).float()

            root_orientation = predicted_angle[..., :3]
            if self.root_orientation_mode == "head_yaw":
                root_orientation = self.root_orientation_from_head_yaw(head_transform, predicted_angle.dtype)
                root_translation = self.root_translation_from_head_position(
                    predicted_angle,
                    root_orientation,
                    head_transform,
                )
            elif self.root_orientation_mode == "head":
                root_orientation, root_translation = self.root_transform_from_head(predicted_angle, head_transform)
            elif self.root_orientation_mode == "model":
                root_translation = self.root_translation_from_head_position(
                    predicted_angle,
                    root_orientation,
                    head_transform,
                )
            else:
                raise ValueError("Unknown root orientation mode: {}".format(self.root_orientation_mode))

            if self.upright_constraint:
                root_orientation = self.constrain_root_upright(root_orientation)
                root_translation = self.root_translation_from_head_position(
                    predicted_angle,
                    root_orientation,
                    head_transform,
                )

            body = self.body_model(
                pose_body=predicted_angle[..., 3:66],
                root_orient=root_orientation,
                trans=root_translation,
            )
            joints = body.Jtr[:, :22, :]
            if self.upright_constraint:
                body, joints = self.constrain_body_output_upright(body, head_transform)

        return AvatarPoseResult(
            frame_index=frame_index,
            joints=joints.detach().cpu().numpy()[0],
            pose_body=predicted_angle[..., 3:66].detach().cpu().numpy()[0],
            root_orient=root_orientation.detach().cpu().numpy()[0],
            translation=root_translation.detach().cpu().numpy()[0],
            body=body,
        )
