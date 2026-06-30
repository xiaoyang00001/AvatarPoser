import numpy as np
import torch

from utils import utils_transform


IDENTITY_TRANSFORM = np.eye(3, dtype=np.float32)
OPENVR_Y_UP_TO_AVATARPOSER_Z_UP = np.array(
    [
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)

COORDINATE_TRANSFORMS = {
    "identity": IDENTITY_TRANSFORM,
    "avatarposer": IDENTITY_TRANSFORM,
    "htc_mr": IDENTITY_TRANSFORM,
    "nolo_openvr": IDENTITY_TRANSFORM,
    "nolo_yup_to_avatarposer": OPENVR_Y_UP_TO_AVATARPOSER_Z_UP,
    "openvr_yup_to_avatarposer": OPENVR_Y_UP_TO_AVATARPOSER_Z_UP,
    "legacy_openvr_yup": OPENVR_Y_UP_TO_AVATARPOSER_Z_UP,
}


def as_numpy(value, dtype=np.float32):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def validate_quaternion(quaternion, tolerance=1e-3):
    quaternion = as_numpy(quaternion)
    if not np.all(np.isfinite(quaternion)):
        raise ValueError("Quaternion contains non-finite values: {}".format(quaternion.tolist()))

    norm = np.linalg.norm(quaternion)
    if abs(norm - 1.0) > tolerance:
        raise ValueError("Quaternion norm is invalid: norm={}, value={}".format(float(norm), quaternion.tolist()))

    return quaternion


def validate_position(position):
    position = as_numpy(position)
    if not np.all(np.isfinite(position)):
        raise ValueError("Position contains non-finite values: {}".format(position.tolist()))
    if np.all(position == 0.0):
        raise ValueError("Position is zero: {}".format(position.tolist()))
    if np.any(np.abs(position) > 100.0):
        raise ValueError("Position is out of range: {}".format(position.tolist()))
    return position


def normalize_quaternion(quaternion):
    quaternion = validate_quaternion(quaternion)
    norm = np.linalg.norm(quaternion)
    return quaternion / norm


def quaternion_to_wxyz(quaternion, order="wxyz"):
    quaternion = normalize_quaternion(quaternion)
    if order == "xyzw":
        x, y, z, w = quaternion
    else:
        w, x, y, z = quaternion
    return np.asarray([w, x, y, z], dtype=np.float32)


def quaternion_to_matrix(quaternion, order="wxyz"):
    w, x, y, z = quaternion_to_wxyz(quaternion, order=order)

    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def matrix_to_quaternion_wxyz(rotation_matrix):
    matrix = as_numpy(rotation_matrix).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        diagonal = np.diag(matrix)
        axis = int(np.argmax(diagonal))
        if axis == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            w = (matrix[2, 1] - matrix[1, 2]) / scale
            x = 0.25 * scale
            y = (matrix[0, 1] + matrix[1, 0]) / scale
            z = (matrix[0, 2] + matrix[2, 0]) / scale
        elif axis == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            w = (matrix[0, 2] - matrix[2, 0]) / scale
            x = (matrix[0, 1] + matrix[1, 0]) / scale
            y = 0.25 * scale
            z = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            w = (matrix[1, 0] - matrix[0, 1]) / scale
            x = (matrix[0, 2] + matrix[2, 0]) / scale
            y = (matrix[1, 2] + matrix[2, 1]) / scale
            z = 0.25 * scale

    return normalize_quaternion(np.asarray([w, x, y, z], dtype=np.float32))


def sixd_to_matrix(rotation_6d):
    d6 = torch.as_tensor(rotation_6d, dtype=torch.float32).reshape(-1, 6)
    return utils_transform.bgs(d6).detach().cpu().numpy().reshape(-1, 3, 3)


def matrix_to_sixd(rotation_matrix):
    rotation_matrix = as_numpy(rotation_matrix).reshape(-1, 3, 3)
    return np.concatenate([rotation_matrix[:, :, 0], rotation_matrix[:, :, 1]], axis=1)


def euler_degrees_to_matrix(euler_degrees):
    rx, ry, rz = np.deg2rad(as_numpy(euler_degrees).reshape(3))
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float32)
    rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
    rot_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return (rot_z @ rot_y @ rot_x).astype(np.float32)


def parse_rotation(pose):
    if "rotation_matrix" in pose:
        return as_numpy(pose["rotation_matrix"]).reshape(3, 3)
    if "matrix" in pose:
        return as_numpy(pose["matrix"]).reshape(3, 3)
    if "rotation_6d" in pose:
        return sixd_to_matrix(pose["rotation_6d"])[0]
    if "quaternion" in pose:
        return quaternion_to_matrix(pose["quaternion"], pose.get("quaternion_order", "xyzw"))
    if "rotation" in pose:
        rotation = as_numpy(pose["rotation"])
        if rotation.size == 9:
            return rotation.reshape(3, 3)
        if rotation.size == 6:
            return sixd_to_matrix(rotation)[0]
        if rotation.size == 4:
            return quaternion_to_matrix(rotation, pose.get("quaternion_order", "xyzw"))
    raise ValueError("Pose must contain rotation_matrix, matrix, rotation_6d, quaternion, or rotation.")


def parse_position(pose):
    if "position" in pose:
        return validate_position(pose["position"]).reshape(3)
    if "pos" in pose:
        return validate_position(pose["pos"]).reshape(3)
    if "translation" in pose:
        return validate_position(pose["translation"]).reshape(3)
    raise ValueError("Pose must contain position, pos, or translation.")


def coordinate_transform_matrix(name):
    if name not in COORDINATE_TRANSFORMS:
        raise ValueError("Unknown tracking coordinate system: {}".format(name))
    return COORDINATE_TRANSFORMS[name]


def coordinate_transform_names():
    return tuple(COORDINATE_TRANSFORMS.keys())


def transform_rotation(rotation_matrix, coordinate_system="identity", mode="pose"):
    basis = coordinate_transform_matrix(coordinate_system)
    rotation_matrix = as_numpy(rotation_matrix).reshape(3, 3)
    if mode == "pose":
        return (basis @ rotation_matrix).astype(np.float32)
    if mode == "basis":
        return (basis @ rotation_matrix @ basis.T).astype(np.float32)
    raise ValueError("Unknown rotation transform mode: {}".format(mode))


def transform_position(position, coordinate_system="identity"):
    basis = coordinate_transform_matrix(coordinate_system)
    position = as_numpy(position).reshape(3)
    return (basis @ position).astype(np.float32)
