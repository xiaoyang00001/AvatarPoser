import torch


def aa2matrot(pose):
    """Convert axis-angle rotations to 3x3 rotation matrices."""
    orig_shape = pose.shape[:-1]
    pose = pose.reshape(-1, 3)
    dtype = pose.dtype
    device = pose.device

    angle = torch.linalg.norm(pose, dim=1, keepdim=True)
    axis = pose / angle.clamp_min(1e-8)
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]

    zeros = torch.zeros_like(x)
    k = torch.stack(
        [
            zeros,
            -z,
            y,
            z,
            zeros,
            -x,
            -y,
            x,
            zeros,
        ],
        dim=1,
    ).reshape(-1, 3, 3)

    eye = torch.eye(3, dtype=dtype, device=device).unsqueeze(0)
    sin = torch.sin(angle).reshape(-1, 1, 1)
    cos = torch.cos(angle).reshape(-1, 1, 1)
    rot = eye + sin * k + (1.0 - cos) * torch.bmm(k, k)

    small = (angle.reshape(-1) < 1e-8).reshape(-1, 1, 1)
    rot = torch.where(small, eye.expand_as(rot), rot)
    return rot.reshape(*orig_shape, 3, 3)


def matrot2aa(pose_matrot):
    """Convert 3x3 rotation matrices to axis-angle rotations."""
    orig_shape = pose_matrot.shape[:-2]
    rot = pose_matrot.reshape(-1, 3, 3)

    trace = rot[:, 0, 0] + rot[:, 1, 1] + rot[:, 2, 2]
    cos_angle = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    angle = torch.acos(cos_angle)

    axis = torch.stack(
        [
            rot[:, 2, 1] - rot[:, 1, 2],
            rot[:, 0, 2] - rot[:, 2, 0],
            rot[:, 1, 0] - rot[:, 0, 1],
        ],
        dim=1,
    )
    axis = axis / (2.0 * torch.sin(angle).unsqueeze(1)).clamp_min(1e-8)
    aa = axis * angle.unsqueeze(1)

    small = angle < 1e-6
    if small.any():
        aa[small] = 0.5 * torch.stack(
            [
                rot[small, 2, 1] - rot[small, 1, 2],
                rot[small, 0, 2] - rot[small, 2, 0],
                rot[small, 1, 0] - rot[small, 0, 1],
            ],
            dim=1,
        )

    return aa.reshape(*orig_shape, 3)


def local2global_pose(local_pose, kintree_table):
    """Propagate local joint rotations through the kinematic tree."""
    local_pose = local_pose.reshape(*local_pose.shape[:-1], 3, 3)
    parents = kintree_table.to(device=local_pose.device, dtype=torch.long).reshape(-1)

    global_pose = [local_pose[:, 0]]
    for joint_idx in range(1, local_pose.shape[1]):
        parent_idx = int(parents[joint_idx].item())
        global_pose.append(torch.matmul(global_pose[parent_idx], local_pose[:, joint_idx]))

    return torch.stack(global_pose, dim=1)
