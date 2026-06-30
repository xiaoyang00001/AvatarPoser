import argparse
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.network import AvatarPoser  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Run a minimal AvatarPoser checkpoint smoke test.")
    parser.add_argument("--checkpoint", default="model_zoo/avatarposer.pth")
    parser.add_argument("--output", default="results/smoke_test_avatarposer.json")
    parser.add_argument("--window-size", type=int, default=40)
    parser.add_argument("--input-dim", type=int, default=54)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = REPO_ROOT / args.checkpoint
    output_path = REPO_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = AvatarPoser(
        input_dim=args.input_dim,
        output_dim=132,
        num_layer=3,
        embed_dim=256,
        nhead=8,
        body_model=None,
        device=device,
    ).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    if isinstance(state_dict, dict) and "params" in state_dict:
        state_dict = state_dict["params"]
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    input_tensor = torch.zeros(1, args.window_size, args.input_dim, device=device)
    with torch.no_grad():
        global_orientation, joint_rotation = model(input_tensor, do_fk=False)

    result = {
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "input_shape": list(input_tensor.shape),
        "global_orientation_shape": list(global_orientation.shape),
        "joint_rotation_shape": list(joint_rotation.shape),
        "global_orientation_mean": float(global_orientation.mean().cpu()),
        "joint_rotation_mean": float(joint_rotation.mean().cpu()),
    }

    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
