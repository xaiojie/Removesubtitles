#!/usr/bin/env python3
"""Install the optional, independently versioned single-stage LTX backend."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile


SCRIPT_DIR = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('ltx_deployment_downloads', SCRIPT_DIR / 'deploy_helper.py')
downloader = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = downloader
spec.loader.exec_module(downloader)


def read_manifest():
    return json.loads((SCRIPT_DIR / 'ltx_manifest.json').read_text(encoding='utf-8'))


def selected_tasks(options):
    return ['subtitle', 'watermark'] if options.tasks == 'both' else [options.tasks]


def selected_assets(manifest, options):
    tasks = set(selected_tasks(options)) | {'shared'}
    return [with_download_urls(item, options) for item in manifest['assets'] if item['task'] in tasks]


def with_download_urls(asset, options):
    adjusted = dict(asset)
    originals = list(asset['urls'])
    mirrors = []
    for url in originals:
        if url.startswith('https://huggingface.co/') and options.hf_endpoint:
            mirrors.append(options.hf_endpoint.rstrip('/') + url[len('https://huggingface.co'):])
        if url.startswith('https://github.com/') and options.github_prefix:
            mirrors.append(options.github_prefix.rstrip('/') + '/' + url)
    preferred = mirrors + originals if options.network == 'cn' else originals + mirrors
    adjusted['urls'] = list(dict.fromkeys(preferred))
    return adjusted


def venv_python(root):
    return Path(root) / '.venv/bin/python'


def safe_source_path(root, relative):
    root = Path(root).resolve()
    relative = PurePosixPath(relative)
    if relative.is_absolute() or '..' in relative.parts or not relative.parts:
        raise ValueError('LTX 源码包含不安全路径。')
    target = root
    for part in relative.parts:
        target = target / part
        if target.is_symlink():
            raise ValueError(f'LTX 源码路径包含符号链接 {target}')
    if root not in target.resolve().parents:
        raise ValueError('LTX 源码路径超出安装目录。')
    return target


def write_marker(root, name, record, extension='complete'):
    marker = safe_source_path(root, '.install/' + name + '.' + extension)
    marker.parent.mkdir(parents=True, exist_ok=True)
    temporary = marker.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, marker)


def clear_markers(root, *names):
    for name in ('runtime', 'test', *names):
        safe_source_path(root, '.install/' + name + '.complete').unlink(missing_ok=True)


def state_record(options, manifest):
    return dict(tasks=selected_tasks(options), source_revision=manifest.get('source', {}).get('revision'),
                manifest_sha256=hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest())


@contextmanager
def install_lock(root):
    state = safe_source_path(root, '.install/install.lock')
    state.parent.mkdir(parents=True, exist_ok=True)
    with state.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('已有 LTX 安装或校验任务正在运行。') from None
        yield


def validate_source(root, manifest):
    for item in manifest['source']['files']:
        path = safe_source_path(root, item['path'])
        if not downloader.valid_file(path, item):
            raise RuntimeError(f'LTX 固定版本源码缺失或校验失败 {path}，请运行 source。')


def extract_source(archive, root, files):
    """Validate every archive member before writing the selected source files."""
    expected = {item['path']: item for item in files}
    with tarfile.open(archive, 'r:gz') as incoming:
        members = {}
        top_level = set()
        for member in incoming.getmembers():
            parts = PurePosixPath(member.name).parts
            if (member.name.startswith('/') or '..' in parts or not parts
                    or not (member.isdir() or member.isfile())):
                raise ValueError('LTX 源码压缩包包含不安全路径或链接。')
            top_level.add(parts[0])
            if len(parts) < 2:
                continue
            relative = '/'.join(parts[1:])
            if relative in expected:
                if relative in members or not member.isfile():
                    raise ValueError('LTX 源码压缩包包含重复或无效文件。')
                safe_source_path(root, relative)
                members[relative] = member
        if len(top_level) != 1 or set(members) != set(expected):
            raise ValueError('LTX 源码压缩包缺少所需文件。')
        # Stage and hash every selected file before modifying existing source.
        Path(root).mkdir(parents=True, exist_ok=True)
        safe_source_path(root, '.install').mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='ltx-source-', dir=Path(root) / '.install') as directory:
            staging = Path(directory)
            for relative, member in members.items():
                target = staging / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if member.size != expected[relative]['size']:
                    raise ValueError(f'LTX 源码文件大小不匹配 {relative}')
                with incoming.extractfile(member) as source, target.open('wb') as output:
                    shutil.copyfileobj(source, output)
                if not downloader.valid_file(target, expected[relative]):
                    raise ValueError(f'LTX 源码文件 SHA-256 不匹配 {relative}')
            for relative in members:
                target = safe_source_path(root, relative)
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staging / relative, target)


def check_space(root, assets):
    needed = 0
    for item in assets:
        path = downloader.asset_path(root, item)
        if downloader.valid_file(path, item):
            continue
        partial = path.with_name(path.name + '.part')
        prefix = partial.stat().st_size if partial.is_file() else 0
        if prefix > item['size']:
            prefix = 0
        released = path.stat().st_blocks * 512 if path.is_file() else 0
        needed += max(0, item['size'] - prefix - released)
    if shutil.disk_usage(root).free < needed + 512 * 1024**2:
        raise RuntimeError(f'LTX 目录空间不足 {root}，还需约 {needed / 1024**3:.1f} GiB，并预留 512 MiB。')


def source(options, manifest):
    clear_markers(options.root, 'source')
    try:
        validate_source(options.root, manifest)
    except RuntimeError:
        archive = with_download_urls(manifest['source']['archive'], options)
        if not options.offline:
            check_space(options.root, [archive])
        downloader.download_asset(archive, options.root, offline=options.offline)
        extract_source(downloader.asset_path(options.root, archive), options.root, manifest['source']['files'])
        validate_source(options.root, manifest)
    write_marker(options.root, 'source', state_record(options, manifest))
    print(f"LTX 源码校验通过 {manifest['source']['revision']}", flush=True)


def environment(options, manifest):
    clear_markers(options.root, 'environment')
    target = options.root / '.venv'
    if options.root.resolve() == SCRIPT_DIR.parent:
        raise RuntimeError('LTX_ROOT 必须使用独立目录，以便创建独立虚拟环境。')
    if target.is_symlink():
        raise RuntimeError('LTX .venv 必须使用独立目录。')
    result = subprocess.run([options.python, '-c', 'import json,sys; print(json.dumps(list(sys.version_info[:2])))'],
                            check=True, capture_output=True, text=True)
    version = tuple(json.loads(result.stdout.strip()))
    if not (3, 10) <= version <= (3, 13):
        raise RuntimeError('LTX 需要 Python 3.10 至 3.13，请通过 --python 选择解释器。')
    if options.offline and (options.wheelhouse is None or not options.wheelhouse.is_dir()):
        raise RuntimeError('LTX 离线环境安装需要 --wheelhouse 指向完整的本地 wheel 目录。')
    if not venv_python(options.root).is_file():
        subprocess.run([options.python, '-m', 'venv', str(target)], check=True)
    python = str(venv_python(options.root))
    python_environment = {**os.environ, 'PYTHONNOUSERSITE': '1'}
    python_environment.pop('PYTHONHOME', None)
    python_environment.pop('PYTHONPATH', None)
    result = subprocess.run([python, '-c', 'import json,sys; print(json.dumps([sys.prefix,sys.base_prefix]))'],
                            check=True, capture_output=True, text=True, env=python_environment)
    prefix, base_prefix = json.loads(result.stdout.strip())
    if Path(prefix).resolve() != target.resolve() or prefix == base_prefix:
        raise RuntimeError('LTX 解释器未使用独立虚拟环境，请检查 .venv。')
    common = ['--no-index', '--find-links', str(options.wheelhouse)] if options.offline else []
    package_source = common or ['--index-url', options.pip_index or (
        'https://pypi.tuna.tsinghua.edu.cn/simple' if options.network == 'cn' else 'https://pypi.org/simple')]
    torch_source = common or ['--index-url', options.torch_index]
    subprocess.run([python, '-m', 'pip', 'install', '--upgrade', *package_source, 'pip==25.2'],
                   check=True, env=python_environment)
    status = subprocess.run([python, '-c', '''
import json
import torch
import torchvision
import torchaudio
print('LTX_TORCH_STATUS ' + json.dumps(dict(cuda=torch.version.cuda,
      torch=torch.__version__.split('+')[0], torchvision=torchvision.__version__.split('+')[0],
      torchaudio=torchaudio.__version__.split('+')[0])))
'''], capture_output=True, text=True, env=python_environment)
    installed = {}
    for line in status.stdout.splitlines():
        if line.startswith('LTX_TORCH_STATUS '):
            installed = json.loads(line[len('LTX_TORCH_STATUS '):])
    expected_cuda = None
    for wheel, version in [('cu126', '12.6'), ('cu128', '12.8')]:
        if options.torch_index.rstrip('/').endswith('/' + wheel):
            expected_cuda = version
    compatible = (status.returncode == 0 and bool(installed.get('cuda'))
                  and installed.get('torch') == '2.8.0' and installed.get('torchvision') == '0.23.0'
                  and installed.get('torchaudio') == '2.8.0'
                  and (expected_cuda is None or installed['cuda'] == expected_cuda))
    # pip treats CPU and different CUDA local builds as satisfying torch==2.8.0.
    # Replace that stack explicitly when its backend is unsuitable.
    replacement = [] if compatible else ['--force-reinstall']
    subprocess.run([python, '-m', 'pip', 'install', *torch_source, *replacement,
                    'torch==2.8.0', 'torchvision==0.23.0', 'torchaudio==2.8.0'],
                   check=True, env=python_environment)
    subprocess.run([python, '-m', 'pip', 'install', *package_source,
                    '-r', str(SCRIPT_DIR / 'requirements-ltx.txt')], check=True, env=python_environment)
    subprocess.run([python, '-m', 'pip', 'check'], check=True, env=python_environment)
    write_marker(options.root, 'environment', state_record(options, manifest))
    print(f'LTX 独立环境安装完成 {python}，请继续运行 verify。', flush=True)


def validate_assets(options, manifest):
    assets = selected_assets(manifest, options)
    for item in assets:
        path = downloader.asset_path(options.root, item)
        if not downloader.valid_file(path, item):
            raise RuntimeError(f'LTX 文件缺失或 SHA-256 校验失败 {path}，请运行 models。')
    return assets


def models(options, manifest):
    clear_markers(options.root, 'models')
    assets = selected_assets(manifest, options)
    if not options.offline:
        check_space(options.root, assets)
    with ThreadPoolExecutor(max_workers=options.workers) as pool:
        futures = [pool.submit(downloader.download_asset, item, options.root, offline=options.offline) for item in assets]
        for future in as_completed(futures):
            future.result()
    record = state_record(options, manifest)
    write_marker(options.root, 'selection', record, extension='json')
    write_marker(options.root, 'models', record)
    print('LTX 所选模型全部通过 SHA-256 校验。', flush=True)


def read_tensor_header(path):
    """Validate safetensors structure without allocating model weights."""
    size = path.stat().st_size
    with path.open('rb') as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f'safetensors 文件头不完整 {path}')
        header_size = struct.unpack('<Q', prefix)[0]
        if header_size < 2 or header_size > min(128 * 1024**2, size - 8):
            raise ValueError(f'safetensors 文件头大小无效 {path}')
        header = json.loads(stream.read(header_size))
    if not isinstance(header, dict):
        raise ValueError(f'safetensors 文件头格式无效 {path}')
    tensors = {key: value for key, value in header.items() if key != '__metadata__'}
    if not tensors:
        raise ValueError(f'safetensors 文件不包含张量 {path}')
    item_sizes = dict(BOOL=1, U8=1, I8=1, F8_E4M3=1, F8_E5M2=1, F8_E4M3FN=1,
                      I16=2, U16=2, F16=2, BF16=2, I32=4, U32=4, F32=4, F64=8, I64=8, U64=8)
    intervals = []
    for value in tensors.values():
        try:
            start, end = value['data_offsets']
            shape = value['shape']
            if (not isinstance(shape, list) or any(type(dimension) is not int or dimension < 0 for dimension in shape)
                    or type(start) is not int or type(end) is not int or start < 0 or end < start
                    or end - start != math.prod(shape) * item_sizes[value['dtype']]):
                raise ValueError('张量大小或偏移无效')
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f'safetensors 张量结构无效 {path}') from error
        intervals.append((start, end))
    position = 0
    for start, end in sorted(intervals):
        if start != position:
            raise ValueError(f'safetensors 张量存在缺失或重叠 {path}')
        position = end
    if position != size - header_size - 8:
        raise ValueError(f'safetensors 文件数据不完整 {path}')
    return header


def validate_model_layout(options, manifest):
    gemma_headers = {}
    for item in selected_assets(manifest, options):
        path = downloader.asset_path(options.root, item)
        if path.suffix == '.json':
            json.loads(path.read_text(encoding='utf-8'))
        if path.suffix != '.safetensors':
            continue
        header = read_tensor_header(path)
        if item['role'] == 'checkpoint':
            prefixes = ['model.diffusion_model.', 'vae.encoder.', 'vae.decoder.', 'audio_vae.',
                        'vocoder.', 'text_embedding_projection.']
            if not all(any(key.startswith(prefix) for key in header) for prefix in prefixes):
                raise ValueError('LTX Insight checkpoint 缺少必要的模型组件。')
            config = json.loads(header.get('__metadata__', {}).get('config', '{}'))
            if config.get('transformer', {}).get('caption_channels') != 3840:
                raise ValueError('LTX checkpoint 与 Gemma 12B 文本维度不匹配。')
        elif item['role'] == 'gemma_weights':
            gemma_headers[path.name] = header
        elif item['role'] == 'lora':
            if not any(key.endswith('.lora_A.weight') for key in header):
                raise ValueError(f'LTX IC-LoRA 张量格式不匹配 {path}')
    gemma_root = options.root / 'models/gemma_configs'
    discovered = {path.relative_to(gemma_root).as_posix() for path in gemma_root.rglob('*.safetensors')}
    if discovered != set(gemma_headers):
        raise ValueError('Gemma 目录包含清单外的 safetensors 文件，请移出旧版或额外的权重。')
    index = json.loads((gemma_root / 'model.safetensors.index.json').read_text())['weight_map']
    indexed = {name: set() for name in gemma_headers}
    for key, filename in index.items():
        if filename not in gemma_headers or key not in gemma_headers[filename] or not key.startswith('model.'):
            raise ValueError('Gemma 分片索引与原生 HF 文本权重不匹配。')
        indexed[filename].add(key)
    for filename, header in gemma_headers.items():
        if indexed[filename] != set(header) - {'__metadata__'}:
            raise ValueError(f'Gemma 分片未被索引完整引用 {filename}')
    config = json.loads((gemma_root / 'config.json').read_text())
    expected = dict(hidden_size=3840, num_hidden_layers=48, num_attention_heads=16,
                    num_key_value_heads=8, head_dim=256, vocab_size=262208)
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError('Gemma 配置与 LTX 上游 12B 文本编码器不匹配。')


RUNTIME_PROBE = r'''
import gc
import importlib.metadata
import io
import json
import os
from pathlib import Path
import sys

root = Path(os.environ['LTX_INSTALL_ROOT'])
if Path(sys.prefix).resolve() != (root / '.venv').resolve():
    raise RuntimeError('LTX interpreter is outside its independent virtual environment')
expected = json.loads(os.environ['LTX_EXPECTED_VERSIONS'])
for package, version in expected.items():
    installed = importlib.metadata.version(package).split('+')[0]
    if installed != version:
        raise RuntimeError(f'{package} {installed} does not match pinned version {version}')
import torch
import torchvision
import torchaudio
import accelerate
import av
import einops
import numpy as np
import scipy
import sentencepiece
from safetensors import safe_open
import transformers
def validate_cuda():
    if not torch.cuda.is_available() or not torch.version.cuda:
        raise RuntimeError('PyTorch cannot use CUDA')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('LTX requires CUDA bfloat16 support')
    x = torch.ones((16, 16), device='cuda', dtype=torch.bfloat16)
    if x.matmul(x)[0, 0].item() != 16:
        raise RuntimeError('CUDA bfloat16 matrix multiplication failed')
    x.to(torch.float8_e4m3fn).to(torch.bfloat16)
    q = torch.ones((1, 1, 8, 32), device='cuda', dtype=torch.bfloat16)
    torch.nn.functional.scaled_dot_product_attention(q, q, q)
    torch.cuda.synchronize()
validate_cuda()
media = io.BytesIO()
with av.open(media, mode='w', format='mp4') as container:
    stream = container.add_stream('libx264', rate=24)
    stream.width = stream.height = 32
    stream.pix_fmt = 'yuv420p'
    for packet in stream.encode(av.VideoFrame.from_ndarray(np.zeros((32, 32, 3), dtype=np.uint8), format='rgb24')):
        container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
media.seek(0)
with av.open(media) as container:
    if len(list(container.decode(video=0))) != 1:
        raise RuntimeError('PyAV H.264 roundtrip failed')
sys.path.insert(0, str(root))
import run_pipeline as sdk
from ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader
from ltx_core.text_encoders.gemma.config import GEMMA3_CONFIG_FOR_LTX
from ltx_core.text_encoders.gemma.encoders.encoder_configurator import GemmaTextEncoderConfigurator
from ltx_pipelines.utils import media_io
paths = sdk.resolve_model_paths()
encoder_builder = sdk.CustomPromptEncoder(paths.checkpoint, paths.gemma_root, torch.bfloat16,
                                          torch.device('cuda'))._text_encoder_builder
# Meta construction exercises the actual Gemma configurator, module operations,
# local tokenizer and image processor while leaving large weights on disk.
encoder = encoder_builder.meta_model({}, encoder_builder.module_ops)
tokens = encoder.tokenizer.tokenize_with_weights('Remove subtitles and text overlays.')
if not tokens.get('gemma') or not any(item[1] for item in tokens['gemma']):
    raise RuntimeError('Gemma local tokenization failed')
if encoder.processor is None:
    raise RuntimeError('Gemma image processor failed to load')
parameters = dict(encoder.named_parameters(remove_duplicate=False))
loaded = set()
for shard in encoder_builder.model_path:
    with safe_open(shard, framework='pt', device='cpu') as weights:
        for key in weights.keys():
            mapped = encoder_builder.model_sd_ops.apply_to_key(key)
            shape = weights.get_slice(key).get_shape()
            value = torch.empty(shape, device='meta')
            for mapped_key, mapped_value in encoder_builder.model_sd_ops.apply_to_key_value(mapped, value):
                if mapped_key not in parameters or list(parameters[mapped_key].shape) != list(mapped_value.shape):
                    raise RuntimeError(f'Gemma weight does not match upstream architecture {key}')
                loaded.add(mapped_key)
needed = {key for key in parameters if key.startswith('model.model.language_model.') or key == 'model.lm_head.weight'}
if not needed.issubset(loaded):
    raise RuntimeError('Gemma text encoder has missing weights')
checkpoint_config = SafetensorsModelStateDictLoader().metadata(paths.checkpoint)
if checkpoint_config.get('transformer', {}).get('caption_channels') != GEMMA3_CONFIG_FOR_LTX.text_config.hidden_size:
    raise RuntimeError('Checkpoint and Gemma dimensions differ')
for task in json.loads(os.environ['LTX_INSTALL_TASKS']):
    settings = sdk.resolve_mode_settings(task + '_rm')
    pipeline = sdk.build_pipeline(settings.lora_path)
    if pipeline['reference_downscale_factor'] < 1:
        raise RuntimeError('Invalid IC-LoRA reference scale')
    del pipeline
del encoder
gc.collect()
torch.cuda.empty_cache()
print('LTX_RUNTIME ' + json.dumps(dict(cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
      packages=expected, tokenization=True, gemma_weight_mapping=True, media_roundtrip=True,
      sample_inference=False)))
'''


def run_runtime_probe(options):
    python = venv_python(options.root)
    if not python.is_file():
        raise RuntimeError(f'LTX 独立解释器缺失 {python}，请运行 env。')
    requirements = {}
    for line in (SCRIPT_DIR / 'requirements-ltx.txt').read_text().splitlines():
        if line.strip() and not line.startswith('#'):
            name, version = line.split('==')
            requirements[name] = version
    requirements.update(torch='2.8.0', torchvision='0.23.0', torchaudio='2.8.0')
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': options.gpu, 'HF_HUB_OFFLINE': '1',
           'TRANSFORMERS_OFFLINE': '1', 'PYTHONNOUSERSITE': '1', 'PYTHONPATH': str(options.root),
           'LTX_INSTALL_ROOT': str(options.root), 'LTX_INSTALL_TASKS': json.dumps(selected_tasks(options)),
           'LTX_EXPECTED_VERSIONS': json.dumps(requirements)}
    env.pop('PYTHONHOME', None)
    try:
        result = subprocess.run([str(python), '-c', RUNTIME_PROBE], cwd=options.root, env=env,
                                check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, timeout=300)
    except subprocess.CalledProcessError as error:
        raise RuntimeError('LTX 运行校验失败\n' + (error.stdout or '')[-6000:]) from error
    for line in reversed(result.stdout.splitlines()):
        if line.startswith('LTX_RUNTIME '):
            return json.loads(line[len('LTX_RUNTIME '):])
    raise RuntimeError('LTX 运行校验没有返回成功结果。')


def verify(options, manifest):
    clear_markers(options.root, 'environment', 'models')
    validate_source(options.root, manifest)
    validate_assets(options, manifest)
    validate_model_layout(options, manifest)
    probe = run_runtime_probe(options)
    record = dict(state_record(options, manifest), probe=probe)
    write_marker(options.root, 'selection', record, extension='json')
    write_marker(options.root, 'models', record)
    write_marker(options.root, 'environment', record)
    write_marker(options.root, 'runtime', record)
    print(f"LTX 运行校验通过，CUDA {probe.get('cuda')}，GPU {probe.get('gpu')}。", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['source', 'env', 'models', 'verify'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--python', default=sys.executable, help='创建 LTX 独立环境的基础解释器')
    parser.add_argument('--network', choices=['cn', 'global'], default='cn')
    parser.add_argument('--hf-endpoint', default=os.environ.get('HF_ENDPOINT'))
    parser.add_argument('--github-prefix', default='')
    parser.add_argument('--workers', type=int, choices=range(1, 9), default=2)
    parser.add_argument('--tasks', choices=['subtitle', 'watermark', 'both'], default='subtitle')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--wheelhouse', type=Path)
    parser.add_argument('--pip-index')
    parser.add_argument('--torch-index', default='https://download.pytorch.org/whl/cu128')
    parser.add_argument('--gpu', default='0')
    result = parser.parse_args(argv)
    result.root = result.root.expanduser().resolve()
    if result.wheelhouse is not None:
        result.wheelhouse = result.wheelhouse.expanduser().resolve()
    if not result.hf_endpoint:
        result.hf_endpoint = 'https://hf-mirror.com' if result.network == 'cn' else 'https://huggingface.co'
    return result


def main(argv=None):
    options = parse_args(argv)
    manifest = read_manifest()
    operations = dict(source=source, env=environment, models=models, verify=verify)
    try:
        with install_lock(options.root):
            operations[options.stage](options, manifest)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f'LTX 安装失败 {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
