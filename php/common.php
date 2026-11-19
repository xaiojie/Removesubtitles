<?php
declare(strict_types=1);

const ROOT = __DIR__ . '/..';
const MODES = [
    'lite' => ['label' => '快速', 'subtitles' => true, 'watermarks' => false],
    'standard' => ['label' => '标准', 'subtitles' => true, 'watermarks' => true],
    'fine' => ['label' => '精细', 'subtitles' => true, 'watermarks' => true],
];

function dataDir(): string
{
    $dir = getenv('SUBTITLE_DATA_DIR') ?: ROOT . '/php/var';
    if (!is_dir($dir) && !mkdir($dir, 0770, true) && !is_dir($dir)) {
        throw new RuntimeException('无法建立任务目录');
    }
    return realpath($dir);
}

function db(): PDO
{
    static $pdo;
    if ($pdo) {
        return $pdo;
    }
    $pdo = new PDO('sqlite:' . dataDir() . '/queue.sqlite', null, null, [PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION]);
    $pdo->exec('PRAGMA busy_timeout=10000');
    $pdo->exec("CREATE TABLE IF NOT EXISTS jobs (
        id TEXT PRIMARY KEY, name TEXT NOT NULL, mode TEXT NOT NULL, state TEXT NOT NULL,
        percent INTEGER NOT NULL DEFAULT 0, stage TEXT NOT NULL, error TEXT, created INTEGER NOT NULL
    )");
    return $pdo;
}

function query(string $sql, array $values = []): PDOStatement
{
    $s = db()->prepare($sql);
    $s->execute($values);
    return $s;
}

function validRegions(mixed $regions, array $size, string $mode): array
{
    if (!is_array($regions) || !array_is_list($regions) || !$regions) {
        throw new InvalidArgumentException('regions 需要至少一个字幕或水印框');
    }
    $normalized = [];
    foreach ($regions as $region) {
        $type = $region->type ?? null;
        $box = $region->roi ?? null;
        if (!in_array($type, ['subtitle', 'watermark'], true) ||
            !is_array($box) || !array_is_list($box) || count($box) !== 4 ||
            count(array_filter($box, 'is_int')) !== 4) {
            throw new InvalidArgumentException('每个区域需要 type 和四个整数 roi 坐标');
        }
        [$x, $y, $r, $b] = $box;
        if ($x < 0 || $y < 0 || $r - $x < 2 || $b - $y < 2 || $r > $size[0] || $b > $size[1]) {
            throw new InvalidArgumentException('选区超出视频显示尺寸或过小');
        }
        if ($mode === 'lite' && $type === 'watermark') {
            throw new InvalidArgumentException('快速档只处理字幕，去水印请选择 standard 或 fine');
        }
        $normalized[] = ['type' => $type, 'roi' => $box];
    }
    return $normalized;
}

function dimensions(string $file): array
{
    $cmd = [getenv('FFPROBE') ?: 'ffprobe', '-v', 'error', '-protocol_whitelist', 'file,pipe',
        '-format_whitelist', 'mov,matroska,webm,avi,flv,mpegts,mpeg,asf,ogg',
        '-select_streams', 'v:0', '-show_entries', 'stream=width,height,sample_aspect_ratio:stream_side_data=rotation',
        '-of', 'json', $file];
    $descriptors = [0 => ['file', '/dev/null', 'r'], 1 => ['pipe', 'w'], 2 => ['file', '/dev/null', 'w']];
    $proc = proc_open($cmd, $descriptors, $pipes);
    if (!is_resource($proc)) {
        throw new RuntimeException('无法启动 ffprobe');
    }
    $info = json_decode(stream_get_contents($pipes[1]), true);
    fclose($pipes[1]);
    $rc = proc_close($proc);
    if ($rc === 127) {
        throw new RuntimeException('未安装 ffprobe');
    }
    $s = $info['streams'][0] ?? [];
    if ($rc || empty($s['width']) || empty($s['height'])) {
        throw new InvalidArgumentException('文件中没有可读取的视频轨道');
    }
    $w = (int)$s['width'];
    $h = (int)$s['height'];
    if (preg_match('~^(\d+):(\d+)$~', $s['sample_aspect_ratio'] ?? '', $m) && (int)$m[1] > 0 && (int)$m[2] > 0) {
        $w = max(1, (int)round($w * (int)$m[1] / (int)$m[2]));
    }
    foreach ($s['side_data_list'] ?? [] as $side) {
        if (abs((int)($side['rotation'] ?? 0)) % 180 === 90) {
            return [$h, $w];
        }
    }
    return [$w, $h];
}
