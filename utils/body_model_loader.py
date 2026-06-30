import torch
import torch.nn as nn
from human_body_prior.body_model.body_model import BodyModel


class BodyModelCompat(nn.Module):
    def __init__(self, body_model, num_betas=16):
        super().__init__()
        self.body_model = body_model
        self.num_betas = num_betas

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.body_model, name)

    def forward(self, root_orient=None, pose_body=None, pose_hand=None, betas=None, trans=None, **kwargs):
        batch_size = None
        for value in (root_orient, pose_body, pose_hand, betas, trans):
            if value is not None:
                batch_size = value.shape[0]
                break
        if batch_size is None:
            batch_size = 1

        device = self.body_model.v_template.device
        dtype = self.body_model.v_template.dtype

        if pose_body is not None:
            device = pose_body.device
            dtype = pose_body.dtype
        elif root_orient is not None:
            device = root_orient.device
            dtype = root_orient.dtype
        elif trans is not None:
            device = trans.device
            dtype = trans.dtype

        model_type = getattr(self.body_model, "model_type", "smplh")
        if root_orient is None:
            root_orient = torch.zeros(batch_size, 3, device=device, dtype=dtype)
        if pose_body is None:
            pose_body = torch.zeros(batch_size, 63, device=device, dtype=dtype)
        if pose_hand is None and model_type in ["smplh", "smplx"]:
            pose_hand = torch.zeros(batch_size, 90, device=device, dtype=dtype)
        if betas is None:
            betas = torch.zeros(batch_size, self.num_betas, device=device, dtype=dtype)
        if trans is None:
            trans = torch.zeros(batch_size, 3, device=device, dtype=dtype)

        return self.body_model(
            root_orient=root_orient,
            pose_body=pose_body,
            pose_hand=pose_hand,
            betas=betas,
            trans=trans,
            **kwargs,
        )


def create_body_model(bm_fname, dmpl_fname=None, num_betas=16, num_dmpls=8):
    """Create a BodyModel across human_body_prior API variants."""
    try:
        return BodyModel(
            bm_fname=bm_fname,
            num_betas=num_betas,
            num_dmpls=num_dmpls,
            dmpl_fname=dmpl_fname,
        )
    except TypeError:
        body_model = BodyModel(
            bm_path=bm_fname,
            model_type="smplh",
            num_betas=num_betas,
        )
        return BodyModelCompat(body_model, num_betas=num_betas)
