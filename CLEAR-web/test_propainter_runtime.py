from contextlib import nullcontext
import sys
from types import ModuleType, SimpleNamespace

from propainter_runtime import ResidentProPainter


def test_resident_runner_strips_python_flags_before_upstream_argparse(tmp_path, monkeypatch):
    source = tmp_path / 'upstream'
    source.mkdir()
    script = source / 'inference_propainter.py'
    script.write_text(
        'import argparse\n'
        'from pathlib import Path\n'
        'parser = argparse.ArgumentParser()\n'
        'parser.add_argument("--video", required=True)\n'
        'args = parser.parse_args()\n'
        'Path(args.video).write_text("executed")\n'
    )
    torch = ModuleType('torch')
    torch.cuda = SimpleNamespace(empty_cache=lambda: None)
    torch.inference_mode = nullcontext
    imageio = ModuleType('imageio')
    imageio.mimwrite = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setitem(sys.modules, 'imageio', imageio)
    for name, attribute in (
        ('model.modules.flow_comp_raft', 'RAFT_bi'),
        ('model.recurrent_flow_completion', 'RecurrentFlowCompleteNet'),
        ('model.propainter', 'InpaintGenerator'),
        ('utils.download_util', 'load_file_from_url'),
    ):
        module = ModuleType(name)
        setattr(module, attribute, lambda *args, **kwargs: None)
        monkeypatch.setitem(sys.modules, name, module)
    marker = tmp_path / 'result'
    original_argv = sys.argv
    runner = ResidentProPainter(source, tmp_path / 'upstream.log')
    runner.run([sys.executable, '-u', str(script), '--video', str(marker)], lambda value: None)
    assert marker.read_text() == 'executed'
    assert sys.argv is original_argv
