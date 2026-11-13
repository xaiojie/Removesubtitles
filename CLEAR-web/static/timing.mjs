// Use progress observed during repair, excluding upload and model loading.
export function estimateRemainingSeconds(first, latest, now) {
  if (!first || !latest) return null;
  const elapsed = (latest.at - first.at) / 1000;
  const progress = latest.percent - first.percent;
  if (elapsed < 5 || progress <= 0 || latest.percent >= 94) return null;
  const sampleAge = (now - latest.at) / 1000;
  const estimate = (94 - latest.percent) * elapsed / progress - sampleAge;
  if (sampleAge > 30 || estimate <= 0) return null;
  return Math.max(1, Math.ceil(estimate));
}
