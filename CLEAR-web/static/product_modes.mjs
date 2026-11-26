// Both single and batch uploads use the same submission fields.
export function setProcessingFields(body, settings) {
  if (settings.mode) {
    body.set('mode', settings.mode);
    body.set('catalog_sha256', settings.catalog_sha256);
  } else {
    body.set('engine', 'auto');
    body.set('inference', JSON.stringify(settings));
  }
}

export function selectedStages(catalog, mode, regions) {
  const kinds = new Set(regions.map(region => region.type));
  return (catalog?.modes[mode]?.stages || [])
    .filter(stage => catalog.stages[stage].region_types.some(kind => kinds.has(kind)));
}

export function supportsRegions(product, regions) {
  return regions.every(region => region.type === 'watermark' ? product.watermarks : product.subtitles);
}
