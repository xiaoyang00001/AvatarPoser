'''
# --------------------------------------------
# utility functions for 3D transformation
# --------------------------------------------
# AvatarPoser: Articulated Full-Body Pose Tracking from Sparse Motion Sensing (ECCV 2022)
# https://github.com/eth-siplab/AvatarPoser
# Jiaxi Jiang (jiaxi.jiang@inf.ethz.ch)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich
'''

import numpy as np
from torch.nn import functional as F
from utils.rotation_tools import aa2matrot, local2global_pose, matrot2aa

import torch


def bgs(d6s):
    d6s = d6s.reshape(-1, 2, 3).permute(0, 2, 1)
    bsz = d6s.shape[0]
    b1 = F.normalize(d6s[:,:,0], p=2, dim=1)
    a2 = d6s[:,:,1]
    c = torch.bmm(b1.view(bsz,1,-1),a2.view(bsz,-1,1)).view(bsz,1)*b1
    b2 = F.normalize(a2-c,p=2,dim=1)
    b3=torch.cross(b1,b2,dim=1)
    return torch.stack([b1,b2,b3],dim=-1)

def matrot2sixd(pose_matrot):
    '''
    :param pose_matrot: Nx3x3
    :return: pose_6d: Nx6
    '''
    pose_6d = torch.cat([pose_matrot[:,:3,0], pose_matrot[:,:3,1]], dim=1)
    return pose_6d


def aa2sixd(pose_aa):
    '''
    :param pose_aa Nx3
    :return: pose_6d: Nx6
    '''
    pose_matrot = aa2matrot(pose_aa)
    pose_6d = matrot2sixd(pose_matrot)
    return pose_6d

def sixd2matrot(pose_6d):
    '''
    :param pose_6d: Nx6
    :return: pose_matrot: Nx3x3
    '''
    rot_vec_1 = pose_6d[:,:3]
    rot_vec_2 = pose_6d[:,3:6]
    rot_vec_3 = torch.cross(rot_vec_1, rot_vec_2, dim=1)
    pose_matrot = torch.stack([rot_vec_1,rot_vec_2,rot_vec_3],dim=-1)
    return pose_matrot

def sixd2aa(pose_6d, batch = False):
    '''
    :param pose_6d: Nx6
    :return: pose_aa: Nx3
    '''
    if batch:
        B,J,C = pose_6d.shape
        pose_6d = pose_6d.reshape(-1,6)
    pose_matrot = sixd2matrot(pose_6d)
    pose_aa = matrot2aa(pose_matrot)
    if batch:
        pose_aa = pose_aa.reshape(B,J,3)
    return pose_aa

def sixd2quat(pose_6d):
    '''
    :param pose_6d: Nx6
    :return: pose_quaternion: Nx4
    '''
    pose_aa = sixd2aa(pose_6d)
    angle = torch.linalg.norm(pose_aa, dim=1, keepdim=True)
    axis = pose_aa / angle.clamp_min(1e-8)
    half_angle = angle * 0.5
    quat = torch.cat([torch.cos(half_angle), axis * torch.sin(half_angle)], dim=1)
    return quat

def quat2aa(pose_quat):
    '''
    :param pose_quat: Nx4
    :return: pose_aa: Nx3
    '''
    quat = F.normalize(pose_quat, p=2, dim=1)
    angle = 2.0 * torch.atan2(torch.linalg.norm(quat[:, 1:], dim=1), quat[:, 0])
    axis = quat[:, 1:] / torch.sin(angle * 0.5).unsqueeze(1).clamp_min(1e-8)
    return axis * angle.unsqueeze(1)
