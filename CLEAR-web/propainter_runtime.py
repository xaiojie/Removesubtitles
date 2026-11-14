"""Reuse upstream ProPainter model instances across windows of one worker job."""
from contextlib import redirect_stderr, redirect_stdout
import gc
import importlib
import os
from pathlib import Path
import re
import runpy
import sys
from urllib.parse import urlsplit


class ProgressLog:
    def __init__(self, stream, progress):
        self.stream = stream
        self.progress = progress

    def write(self, text):
        self.stream.write(text)
        match = re.search(r'(?<!\d)(\d{1,3})%\|', text)
        if match:
            self.progress(min(100, int(match.group(1))))
        return len(text)

    def flush(self):
        self.stream.flush()

    def isatty(self):
        return False


class ResidentProPainter:
    def __init__(self, source, log_path):
        self.source = Path(source).resolve()
        self.log_path = log_path
        self.models = {}

    def constructor(self, original):
        def cached(*arguments, **keywords):
            key = (original.__module__, original.__name__, repr(arguments), repr(sorted(keywords.items())))
            if key not in self.models:
                self.models[key] = original(*arguments, **keywords)
            return self.models[key]
        return cached

    def local_weight(self, url, model_dir=None, progress=True, file_name=None, **kwargs):
        filename = file_name or Path(urlsplit(url).path).name
        path = self.source / 'weights' / filename
        if not path.is_file():
            raise RuntimeError(f'ProPainter 本地模型缺失，请先完成模型下载。{filename}')
        return str(path)

    def run(self, command, progress):
        import torch
        import imageio

        source_path = str(self.source)
        sys.path.insert(0, source_path)
        constructors = (
            ('model.modules.flow_comp_raft', 'RAFT_bi'),
            ('model.recurrent_flow_completion', 'RecurrentFlowCompleteNet'),
            ('model.propainter', 'InpaintGenerator'),
        )
        replacements = []
        previous_argv, previous_directory = sys.argv, Path.cwd()
        try:
            for module_name, attribute in constructors:
                module = importlib.import_module(module_name)
                original = getattr(module, attribute)
                replacements.append((module, attribute, original))
                setattr(module, attribute, self.constructor(original))
            downloader = importlib.import_module('utils.download_util')
            replacements.extend(((downloader, 'load_file_from_url', downloader.load_file_from_url),
                                 (imageio, 'mimwrite', imageio.mimwrite),
                                 (torch.cuda, 'empty_cache', torch.cuda.empty_cache)))
            downloader.load_file_from_url = self.local_weight
            imageio.mimwrite = lambda *args, **kwargs: None
            torch.cuda.empty_cache = lambda: None
            sys.argv = command[1:]
            os.chdir(self.source)
            with self.log_path.open('a', encoding='utf-8') as stream:
                output = ProgressLog(stream, progress)
                with redirect_stdout(output), redirect_stderr(output), torch.inference_mode():
                    runpy.run_path(str(self.source / 'inference_propainter.py'), run_name='__main__')
        finally:
            for module, attribute, original in reversed(replacements):
                setattr(module, attribute, original)
            sys.argv = previous_argv
            os.chdir(previous_directory)
            sys.path.remove(source_path)
            gc.collect()
            torch.cuda.empty_cache()
        progress(100)
