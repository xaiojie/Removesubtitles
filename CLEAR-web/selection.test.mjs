import { test } from 'node:test';
import assert from 'node:assert/strict';
import { selection } from './static/selection.mjs';

test('reverse drag and out-of-bounds pointer map to the video pixels', () => {
  assert.deepEqual(selection([.75, .8], [-.2, .5], 1920, 1080), [0, 540, 1440, 864]);
});
test('a click or sub-pixel drag does not select a region', () => {
  assert.equal(selection([.5, .5], [.501, .501], 640, 360), null);
});
