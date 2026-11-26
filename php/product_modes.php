<?php
declare(strict_types=1);

function productCatalog(): array
{
    static $catalog;
    if ($catalog === null) {
        $source = file_get_contents(__DIR__ . '/product_modes.json');
        if ($source === false) {
            throw new RuntimeException('无法读取处理档位');
        }
        $catalog = json_decode($source, true, flags: JSON_THROW_ON_ERROR);
        $catalog['catalog_sha256'] = hash('sha256', $source);
        $catalog['order'] = array_keys($catalog['modes']);
    }
    return $catalog;
}

function resolveProductPlan(string $mode, array $regionTypes): array
{
    $catalog = productCatalog();
    $product = $catalog['modes'][$mode] ?? null;
    if (!$product) {
        throw new InvalidArgumentException('mode 可选 ' . implode('、', array_keys($catalog['modes'])));
    }
    if (!$regionTypes || array_diff($regionTypes, ['subtitle', 'watermark'])) {
        throw new InvalidArgumentException('需要字幕或水印选区');
    }
    if (in_array('subtitle', $regionTypes, true) && !$product['subtitles']) {
        throw new InvalidArgumentException($product['label'] . ' 不支持字幕选区');
    }
    if (in_array('watermark', $regionTypes, true) && !$product['watermarks']) {
        throw new InvalidArgumentException($product['label'] . ' 只处理字幕，去水印请选择 Pro 或 Max');
    }
    $stages = [];
    $parameters = [];
    foreach ($product['stages'] as $id) {
        $stage = $catalog['stages'][$id];
        if (array_intersect($stage['region_types'], $regionTypes)) {
            $stages[] = $id;
            $key = $stage['parameters'];
            $parameters[$key] = array_replace($catalog['parameters'][$key], $product['overrides'][$key] ?? []);
        }
    }
    return ['catalog_version' => $catalog['catalog_version'], 'catalog_sha256' => $catalog['catalog_sha256'],
        'mode' => $mode, 'stages' => $stages, 'parameters' => $parameters];
}

// Benchmarks use the same resolver as the HTTP endpoint.
if (PHP_SAPI === 'cli' && realpath($_SERVER['SCRIPT_FILENAME']) === __FILE__) {
    try {
        echo json_encode(resolveProductPlan($argv[1] ?? '', array_slice($argv, 2)),
            JSON_THROW_ON_ERROR | JSON_PRESERVE_ZERO_FRACTION | JSON_UNESCAPED_UNICODE), "\n";
    } catch (Throwable $error) {
        fwrite(STDERR, $error->getMessage() . "\n");
        exit(1);
    }
}
