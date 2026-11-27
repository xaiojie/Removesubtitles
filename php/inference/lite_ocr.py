"""Run the frozen PP-OCR detector graph on CUDA with the original Paddle pre/postprocessing."""
import hashlib
import os
from pathlib import Path

import numpy as np
from job_trace import span


def cuda_detector(detector, model_dir, *, split_shapes=False):
    import onnxruntime as ort

    graph = Path(os.environ.get('LITE_OCR_ONNX') or model_dir / 'inference.onnx')
    if not graph.is_file():
        raise ValueError('缺少 GPU 字幕定位模型，请先从已安装的 PP-OCRv5 权重导出 inference.onnx')
    checksum = hashlib.sha256(graph.read_bytes()).hexdigest()
    ort.preload_dlls(directory='')
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 3

    def open_session(slot):
        with span('lite_ocr_graph_load', device='cuda', backend='onnxruntime', graph_sha256=checksum, slot=slot):
            session = ort.InferenceSession(str(graph), sess_options=options, providers=[
                ('CUDAExecutionProvider', {'use_tf32': '0', 'cudnn_conv_algo_search': 'HEURISTIC'})])
            if session.get_providers()[0] != 'CUDAExecutionProvider':
                raise RuntimeError('GPU 字幕定位未能加载 CUDA 执行器')
            return session

    primary = open_session(0)
    input_name = primary.get_inputs()[0].name
    primary_shape = None
    secondary = None

    def infer(x):
        nonlocal primary_shape, secondary
        batch = np.ascontiguousarray(x[0])
        if primary_shape is None:
            primary_shape = batch.shape
        # Switching convolution shapes repeatedly is expensive on the L20.
        # Preserve the first shape and share one additional session among every other size.
        other = split_shapes and batch.shape != primary_shape
        if other and secondary is None:
            secondary = open_session(1)
        session = secondary if other else primary
        with span('lite_ocr_inference', device='cuda', slot=int(other)):
            return session.run(None, {input_name: batch})

    # Keep the pinned PaddleX detector's resizing, normalization, DB thresholds and polygon extraction.
    detector.paddlex_predictor.infer = infer
    detector.inference_device = 'cuda'
    return detector
