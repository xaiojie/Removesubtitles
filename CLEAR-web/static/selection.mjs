export function selection(start, end, width, height) {
  const clamp = (value) => Math.max(0, Math.min(1, value));
  const x = [start[0], end[0]].map(v => clamp(v) * width).sort((a, b) => a - b);
  const y = [start[1], end[1]].map(v => clamp(v) * height).sort((a, b) => a - b);
  if (x[1] - x[0] < 2 || y[1] - y[0] < 2) return null;
  return [Math.floor(x[0]), Math.floor(y[0]), Math.ceil(x[1]), Math.ceil(y[1])];
}
