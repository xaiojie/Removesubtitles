"""Shared CLEAR presets for new jobs, API clients and the Web interface."""
import os


_COMMON = dict(
    steps=5, cfg_scale=1.0, lora_scale=1.0, seed=20261113,
    chunk_frames=33, overlap_frames=4, resolution='native', export_crf=10,
    blend_mode='precise', segment_mode='shots', vae_tiled=False,
    adaptive_processing=False, resident_models=True,
)
_PROFILES = {
    'fast': dict(label='快速', description='局部原像素输入，适合优先处理速度。',
                 options=dict(context='roi', lora_scale=1.2)),
    'balanced': dict(label='较大上下文', description='按画面比例扩展字幕周边，兼顾速度与背景衔接。',
                     options=dict(context='expanded')),
    'quality': dict(label='强化修复', description='12 步全画面处理，更充分覆盖字幕与底板，局部重建可能变模糊。',
                    options=dict(context='frame', steps=12, resolution='auto')),
}


def profile_defaults(name):
    """Return detached inference options. Explicit unknown profile names are errors."""
    if not isinstance(name, str) or name not in _PROFILES:
        raise ValueError('preset 必须是 fast、balanced、quality 或 custom。')
    return dict(_COMMON, **_PROFILES[name]['options'], preset=name)


def profile_catalog():
    return [dict(id=name, label=profile['label'], description=profile['description'],
                 options=profile_defaults(name)) for name, profile in _PROFILES.items()]


def default_profile():
    """Invalid or empty deployment defaults fall back to the balanced preset."""
    value = os.environ.get('CLEAR_PRESET', 'balanced').strip().lower()
    return value if value in _PROFILES else 'balanced'
