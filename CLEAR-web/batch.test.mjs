import { test } from 'node:test';
import assert from 'node:assert/strict';
import { saveVideoDraft } from './static/batch.mjs';

test('each video keeps independent regions and model parameters', () => {
  const first = {}, second = {};
  const regions = [{ type: 'subtitle', roi: [10, 20, 100, 80] }];
  const inference = { subtitle_model: 'clear', clear: { steps: 5 } };
  saveVideoDraft(first, regions, [640, 360], inference, { model: 'clear' });
  regions[0].roi[0] = 50;
  inference.subtitle_model = 'ltx';
  saveVideoDraft(second, regions, [720, 1280], inference, { model: 'ltx' });
  assert.deepEqual(first.regions[0].roi, [10, 20, 100, 80]);
  assert.equal(first.inference.subtitle_model, 'clear');
  assert.deepEqual(second.regions[0].roi, [50, 20, 100, 80]);
  assert.equal(second.inference.subtitle_model, 'ltx');
  assert.ok(first.ready && second.ready);
});

test('a video without its own selection stays unready', () => {
  const item = {};
  saveVideoDraft(item, [], [640, 360], {}, {});
  assert.equal(item.ready, false);
});
