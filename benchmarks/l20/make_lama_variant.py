"""Fold eligible batch-normalization layers into the original FP32 LaMa convolutions."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory


def digest(path):
    with path.open("rb") as reader:
        sha = hashlib.sha256()
        for block in iter(lambda: reader.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def create(source, output):
    import torch

    if output.suffix != ".pt" or output.exists():
        raise ValueError("请给派生权重指定一个新的 .pt 路径")
    frozen = Path(__file__).parent / "results/20261127-round2/lite-final-candidate.json"
    expected = json.loads(frozen.read_text())["weights"]["video-subtitle-remover/backend/models/big-lama/big-lama.pt"]
    if digest(source) != expected:
        raise ValueError("原始 LaMa 权重摘要不匹配")
    original = torch.jit.load(str(source), map_location="cpu").eval()
    variant = torch.jit.freeze(original, optimize_numerics=True)
    bn = [sum(n.kind() == "aten::batch_norm" for n in m.inlined_graph.nodes()) for m in (original, variant)]
    if bn[1] >= bn[0]:
        raise RuntimeError("没有可合并的归一化层")
    metadata = dict(method="fp32_jit_freeze", source_sha256=expected, torch=str(torch.__version__),
                    batch_norm_before=bn[0], batch_norm_after=bn[1], training=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temporary, torch.inference_mode():
        path = Path(temporary) / "fused.pt"
        torch.jit.save(variant, str(path), _extra_files={"lite_variant.json": json.dumps(metadata)})
        reloaded = torch.jit.load(str(path), map_location="cpu").eval()
        generator = torch.Generator().manual_seed(20261127)
        errors = []
        for h, w in ((64, 128), (88, 192)):
            image = torch.rand(1, 3, h, w, generator=generator)
            mask = torch.zeros(1, 1, h, w, dtype=torch.long)
            mask[:, :, h//4:3*h//4, w//6:5*w//6] = 1
            deviation = (reloaded(image, mask) - original(image, mask)).abs()
            errors.append(float(deviation.max()))
            if not torch.isfinite(deviation).all() or errors[-1] > 1/255:
                raise RuntimeError("重新载入的权重与原 FP32 输出差异超过一个颜色级")
        os.link(path, output)
    print(json.dumps(dict(metadata, sha256=digest(output), max_abs_unit_rgb=errors), indent=2))


if __name__ == "__main__":
    import torch
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("source", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    torch.set_num_threads(2)
    create(args.source, args.output)
