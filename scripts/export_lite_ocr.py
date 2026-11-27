"""Build the Lite CUDA detector from the installed Paddle weights, with a resumable provenance record."""
import argparse
import fcntl
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(directory):
    graph = directory/'inference.onnx'
    record = directory/'inference.onnx.json'
    inputs = {name:sha(directory/name) for name in ('inference.json','inference.pdiparams')}
    definition = dict(inputs=inputs,converter=version('paddle2onnx'),opset=17,optimization='None')
    with (directory/'.onnx-export.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if record.is_file() and graph.is_file():
            try:
                previous = json.loads(record.read_text())
            except (OSError,ValueError):
                previous = {}
            if previous.get('definition') == definition and previous.get('sha256') == sha(graph):
                print('已校验 GPU 字幕定位模型',graph)
                return
        with TemporaryDirectory(dir=directory) as temporary:
            out = Path(temporary)/graph.name
            subprocess.run([sys.executable,'-m','paddle2onnx.command','--model_dir',str(directory),
                            '--model_filename','inference.json','--params_filename','inference.pdiparams',
                            '--save_file',str(out),'--opset_version','17','--optimize_tool','None'],check=True)
            import onnx
            onnx.checker.check_model(str(out))
            stamp = Path(temporary)/record.name
            stamp.write_text(json.dumps(dict(definition=definition,sha256=sha(out)),indent=2)+'\n')
            os.replace(out,graph)
            os.replace(stamp,record)
        print('已导出 GPU 字幕定位模型',graph)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('directory',type=Path,help='PP-OCRv5 Mobile 的 ch_det_fast 模型目录')
    export(ap.parse_args().directory.resolve(strict=True))
