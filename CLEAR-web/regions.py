import json

REGION_TYPES = ('subtitle', 'watermark')


def validate_dimensions(dimensions):
    if (not isinstance(dimensions, list) or len(dimensions) != 2
            or any(type(value) is not int or value <= 0 for value in dimensions)):
        raise ValueError('视频尺寸无效。')
    return dimensions


def validate_roi(roi, dimensions):
    if not isinstance(roi, list) or len(roi) != 4 or any(type(value) is not int for value in roi):
        raise ValueError('选区坐标无效。')
    x1, y1, x2, y2 = roi
    width, height = dimensions
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height and x2 - x1 >= 2 and y2 - y1 >= 2):
        raise ValueError('选区超出视频画面或过小。')
    return roi.copy()


def region(raw_roi, raw_dimensions):
    try:
        dimensions = validate_dimensions(json.loads(raw_dimensions))
        roi = json.loads(raw_roi)
    except (ValueError, TypeError):
        raise ValueError('字幕区域或视频尺寸无效。') from None
    return validate_roi(roi, dimensions), dimensions


def validate_regions(values, dimensions):
    validate_dimensions(dimensions)
    if not isinstance(values, list) or not values:
        raise ValueError('请至少框选一处字幕或水印。')
    result = []
    for value in values:
        if not isinstance(value, dict) or set(value) != {'type', 'roi'}:
            raise ValueError('每个选区需要 type 和 roi。')
        if value['type'] not in REGION_TYPES:
            raise ValueError('选区类型必须为 subtitle 或 watermark。')
        result.append(dict(type=value['type'], roi=validate_roi(value['roi'], dimensions)))
    return result


def parse_regions(raw_regions, raw_roi, raw_dimensions):
    if raw_regions is None:
        roi, dimensions = region(raw_roi, raw_dimensions)
        return [dict(type='subtitle', roi=roi)], dimensions
    try:
        dimensions = validate_dimensions(json.loads(raw_dimensions))
        values = json.loads(raw_regions)
    except (ValueError, TypeError):
        raise ValueError('选区列表或视频尺寸无效。') from None
    return validate_regions(values, dimensions), dimensions


def regions_from_config(config):
    values = config.get('regions')
    if values is None:
        values = [dict(type='subtitle', roi=config['roi'])]
    return validate_regions(values, config['dimensions'])


def bounds(regions):
    if not regions:
        raise ValueError('选区列表不能为空。')
    rois = [value['roi'] for value in regions]
    return [min(roi[0] for roi in rois), min(roi[1] for roi in rois),
            max(roi[2] for roi in rois), max(roi[3] for roi in rois)]
