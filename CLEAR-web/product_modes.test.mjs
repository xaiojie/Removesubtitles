import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { selectedStages, setProcessingFields, supportsRegions } from './static/product_modes.mjs';
import { saveVideoDraft } from './static/batch.mjs';

const catalog = JSON.parse(fs.readFileSync(new URL('../php/product_modes.json', import.meta.url)));
const sub = { type: 'subtitle', roi: [1, 6, 20, 14] };
const watermark = { type: 'watermark', roi: [20, 4, 28, 12] };

test('single and batch settings retain each selected catalog mode and send no custom overrides', () => {
  const first = {}, next = {};
  for (const [item, mode] of [[first, 'lite'], [next, 'max']]) {
    saveVideoDraft(item, [sub], [32, 16], { mode, catalog_sha256: 'version' }, { model: mode });
  }
  for (const item of [first, next]) {
    const fields = new FormData();
    setProcessingFields(fields, item.inference);
    assert.deepEqual(Object.fromEntries(fields), { mode: item.form.model, catalog_sha256: 'version' });
  }
  const experimental = new FormData();
  setProcessingFields(experimental, { subtitle_model: 'ltx' });
  assert.deepEqual(Object.fromEntries(experimental), { engine: 'auto', inference: '{"subtitle_model":"ltx"}' });
});

test('catalog controls the displayed stages and the Lite watermark restriction', () => {
  assert.deepEqual(selectedStages(catalog, 'max', [watermark]), ['propainter', 'selective_review']);
  assert.deepEqual(selectedStages(catalog, 'max', [watermark, sub]), ['clear', 'propainter', 'selective_review']);
  assert.equal(supportsRegions(catalog.modes.lite, [sub]), true);
  assert.equal(supportsRegions(catalog.modes.lite, [watermark, sub]), false);
});
