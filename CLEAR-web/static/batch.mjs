export function saveVideoDraft(item, regions, dimensions, inference, form) {
  item.regions = structuredClone(regions);
  item.dimensions = dimensions.slice();
  item.inference = inference ? structuredClone(inference) : null;
  item.form = structuredClone(form);
  item.ready = item.regions.length > 0 && dimensions.every(value => value > 0) && item.inference !== null;
}
