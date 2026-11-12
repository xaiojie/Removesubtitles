from download_remaining import ASSETS, WAN, check
from app import ROOT

WEIGHTS = [
    *(asset[:3] for asset in ASSETS.values()),
    (WAN / 'fun-control.safetensors', 3129305312,
     '4d51c942b0cfe92821c29b37fdb062756e026280084746ba1cc5d5dba39ee684'),
    (WAN / 'Wan2.1_VAE.pth', 507609880,
     '38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981'),
]

for path, size, sha in WEIGHTS:
    check(path, size, sha)
    print(f'{path.name}: SHA-256 OK', flush=True)

(ROOT / 'models.ready').write_text('SHA-256 verified\n', encoding='utf-8')
