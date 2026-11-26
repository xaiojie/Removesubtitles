"""Create an independently loadable CLEAR adapter with rank-truncated updates."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory


def reduced_factors(a, b, rank):
    """Best rank-k approximation to B@A without forming the dense layer update."""
    import torch

    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1] or not 0 < rank < a.shape[0]:
        raise ValueError(f"Invalid factor shapes or rank: {tuple(a.shape)}, {tuple(b.shape)}, {rank}")
    qb, rb = torch.linalg.qr(b.float(), mode="reduced")
    qa, ra = torch.linalg.qr(a.T.float(), mode="reduced")
    left, values, right = torch.linalg.svd(rb @ ra.T, full_matrices=False)
    if not torch.isfinite(values).all() or values[0] == 0:
        raise ValueError("Invalid or empty LoRA update")
    # Keep the existing adapter tensor shapes and alpha convention for the runtime.
    a_out, b_out = torch.zeros_like(a), torch.zeros_like(b)
    scale = values[:rank].sqrt()
    a_out[:rank] = scale[:, None] * (right[:rank] @ qa.T)
    b_out[:, :rank] = (qb @ left[:, :rank]) * scale
    kept = (values[:rank].square().sum() / values.square().sum()).item()
    return a_out.contiguous(), b_out.contiguous(), kept


def create(source, output, rank):
    import torch
    from safetensors.torch import save_file

    if output.suffix != ".safetensors" or output.exists():
        raise ValueError("请指定一个尚不存在的 .safetensors 文件")
    manifest = Path(__file__).resolve().parents[2] / "scripts/model_manifest.json"
    expected = next(row["sha256"] for row in json.loads(manifest.read_text())["assets"]
                    if row["path"].endswith("CLEAR-mask-free-subtitle-removal.pt"))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != expected:
        raise ValueError("源权重与仓库记录的 CLEAR 原始权重不匹配")
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    weights = checkpoint.get("model_state_dict", checkpoint.get("dit_state_dict", checkpoint))
    left = sorted(k for k in weights if ".lora_A." in k)
    right = {k for k in weights if ".lora_B." in k}
    if not left or {k.replace(".lora_A.", ".lora_B.") for k in left} != right:
        raise ValueError("权重中缺少配对的 LoRA A/B 矩阵")
    result = dict(weights)
    energies = []
    for key in left:
        partner = key.replace(".lora_A.", ".lora_B.")
        result[key], result[partner], kept = reduced_factors(weights[key], weights[partner], rank)
        energies.append(kept)
    metadata = {"method": "truncated_svd_of_delta", "effective_rank": str(rank), "source_sha256": digest,
                "layers": str(len(left)), "mean_retained_energy": str(sum(energies) / len(energies))}
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temporary:
        path = Path(temporary) / "variant.safetensors"
        save_file(result, str(path), metadata=metadata)
        os.link(path, output)
    print(json.dumps(dict(metadata, sha256=hashlib.sha256(output.read_bytes()).hexdigest()), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--rank", type=int, default=48)
    args = parser.parse_args()
    create(args.source, args.output, args.rank)
