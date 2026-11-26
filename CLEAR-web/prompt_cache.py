"""Share the exact positive Wan embedding between independent subtitle jobs."""
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile


def embedding_file(directory, encoder, tokenizer, prompt, hardware):
    # Identify the actual local files, including replacements at the same path, without rereading 11 GB.
    assets = []
    for path in [encoder, *sorted(p for p in tokenizer.iterdir() if p.is_file())]:
        stat = path.stat()
        assets.append((path.name, stat.st_dev, stat.st_ino, stat.st_size, stat.st_ctime_ns, stat.st_mtime_ns))
    packages = {name: version(name) for name in ('diffsynth', 'torch', 'tokenizers', 'transformers')}
    identity = json.dumps(['wan-positive-bf16-v1', assets, packages, hardware, prompt], sort_keys=True)
    return directory / (hashlib.sha256(identity.encode()).hexdigest() + '.safetensors')


def read_embedding(file):
    from safetensors import SafetensorError
    from safetensors.torch import load_file
    import torch
    try:
        value = load_file(str(file), device='cpu')['context']
    except (OSError, KeyError, SafetensorError):
        return None
    if value.shape != (1, 512, 4096) or value.dtype != torch.bfloat16 or not torch.isfinite(value).all():
        return None
    return value


def save_embedding(file, tensor):
    from safetensors.torch import save_file
    temporary = None
    try:
        file.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=file.parent, prefix='.embedding-', delete=False) as out:
            temporary = Path(out.name)
        save_file({'context': tensor.detach().cpu().contiguous()}, str(temporary))
        os.replace(temporary, file)
    except OSError as error:
        print(f'CLEAR prompt cache: {error}', flush=True)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
