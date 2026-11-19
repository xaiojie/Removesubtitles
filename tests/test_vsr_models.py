import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('vsr_downloads', ROOT/'video-subtitle-remover/deploy/download_models.py')
downloads = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(downloads)


class VsrModelsTest(unittest.TestCase):
    def test_partial_installs_match_inference_dependencies(self):
        manifest = json.loads((downloads.HERE/'models.json').read_text())
        auto = downloads.selected_files(manifest, 'sttn-auto', 'server')
        self.assertEqual(list(auto), ['sttn-auto/infer_model.pth'])
        paths = downloads.selected_files(manifest, 'propainter', 'mobile')
        self.assertIn('big-lama/big-lama.pt', paths)  # fall back to LaMa on isolated frames
        self.assertIn('propainter/ProPainter.pth', paths)
        self.assertIn('V5/ch_det_fast/inference.pdiparams', paths)
        self.assertNotIn('V5/ch_det/inference.pdiparams', paths)

    def test_assemble_verify_and_reject_corruption_without_replacing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, dest = Path(tmp)/'origin', Path(tmp)/'models'
            src.mkdir()
            parts = []
            for i, data in enumerate([b'first', b'second']):
                name = str(i)
                (src/name).write_bytes(data)
                oid = hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()
                parts.append(dict(path=name, size=len(data), git_blob=oid))
            files = {'sub/model.pt': parts}
            bases = [src.as_uri()]
            downloads.install(dest, files, bases)
            model = dest/'sub/model.pt'
            self.assertEqual(model.read_bytes(), b'firstsecond')
            downloads.install(dest, files, [], verify_only=True)
            model.write_bytes(b'old version')
            (src/'1').write_bytes(b'broken')
            with self.assertRaises(ValueError):
                downloads.install(dest, files, [], verify_only=True)
            with self.assertRaises(ValueError):
                downloads.install(dest, files, bases)
            self.assertEqual(model.read_bytes(), b'old version')
            self.assertFalse(list(dest.rglob('*.part')))


if __name__ == '__main__':
    unittest.main()
