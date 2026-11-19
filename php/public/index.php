<?php
declare(strict_types=1);
require dirname(__DIR__) . '/common.php';

function respond(int $code, array $value): never
{
    http_response_code($code);
    header('Content-Type: application/json; charset=utf-8');
    echo json_encode($value, JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES | JSON_THROW_ON_ERROR);
    exit;
}

function submit(): never
{
    $file = $_FILES['video'] ?? null;
    if (($file['error'] ?? null) !== UPLOAD_ERR_OK) {
        // When the entire POST is too large, PHP discards both form fields and files.
        $limit = trim(ini_get('post_max_size'));
        $scale = ['g' => 1024 ** 3, 'm' => 1024 ** 2, 'k' => 1024];
        $bytes = (int)$limit * ($scale[strtolower(substr($limit, -1))] ?? 1);
        $tooLarge = ($bytes > 0 && (int)($_SERVER['CONTENT_LENGTH'] ?? 0) > $bytes)
            || in_array($file['error'] ?? null, [UPLOAD_ERR_INI_SIZE, UPLOAD_ERR_FORM_SIZE], true);
        respond($tooLarge ? 413 : 400, [
            'error' => $tooLarge ? '上传超出大小限制' : '需要上传完整的 video 文件',
        ]);
    }
    $tmp = $file['tmp_name'];
    if (!is_string($tmp) || !is_uploaded_file($tmp)) {
        respond(400, ['error' => '无效上传']);
    }
    $mode = $_POST['mode'] ?? 'standard';
    if (!is_string($mode) || !isset(MODES[$mode])) {
        respond(422, ['error' => 'mode 可选 lite、standard、fine']);
    }
    $raw = $_POST['regions'] ?? '';
    if (!is_string($raw)) {
        respond(422, ['error' => 'regions 需要 JSON 数组']);
    }
    $size = dimensions($tmp);
    $regions = validRegions(json_decode($raw), $size, $mode);
    $name = basename(str_replace('\\', '/', (string)$file['name']));
    $name = preg_replace('/[[:cntrl:]]/u', '', $name) ?: 'video';
    $id = bin2hex(random_bytes(16));
    $dir = dataDir() . '/' . $id;
    if (!mkdir($dir, 0770)) {
        throw new RuntimeException('无法创建任务');
    }
    try {
        if (!move_uploaded_file($tmp, "$dir/source.video")) {
            throw new RuntimeException('保存上传文件失败');
        }
        $config = ['mode' => $mode, 'dimensions' => $size, 'regions' => $regions, 'original_filename' => $name];
        $json = json_encode($config, JSON_THROW_ON_ERROR | JSON_UNESCAPED_UNICODE);
        if (file_put_contents("$dir/request.json", $json) === false) {
            throw new RuntimeException('保存任务信息失败');
        }
        query("INSERT INTO jobs (id, name, mode, state, stage, created) VALUES (?, ?, ?, 'queued', '排队中', ?)",
            [$id, $name, $mode, time()]);
    } catch (Throwable $e) {
        foreach (['source.video', 'request.json'] as $f) {
            if (is_file("$dir/$f")) {
                unlink("$dir/$f");
            }
        }
        rmdir($dir);
        throw $e;
    }
    header("Location: /api/jobs/$id");
    respond(202, [
        'id' => $id, 'mode' => $mode, 'state' => 'queued', 'dimensions' => $size, 'status_url' => "/api/jobs/$id",
    ]);
}

try {
    header('Cache-Control: no-store');
    $path = parse_url($_SERVER['REQUEST_URI'], PHP_URL_PATH);
    $method = $_SERVER['REQUEST_METHOD'];
    $key = getenv('SUBTITLE_API_TOKEN');
    if ($key !== false && $key !== '' && !hash_equals('Bearer ' . $key, $_SERVER['HTTP_AUTHORIZATION'] ?? '')) {
        header('WWW-Authenticate: Bearer');
        respond(401, ['error' => '需要有效的 Bearer token']);
    }
    if ($path === '/api/modes' && $method === 'GET') {
        respond(200, ['default' => 'standard', 'modes' => MODES]);
    }
    if ($path === '/api/jobs' && $method === 'POST') {
        submit();
    }
    if (!preg_match('~^/api/jobs/([a-f0-9]{32})(/output|/retry)?$~D', $path, $m)) {
        respond(404, ['error' => '接口不存在']);
    }
    $id = $m[1];
    $job = query('SELECT * FROM jobs WHERE id=?', [$id])->fetch(PDO::FETCH_ASSOC);
    if (!$job) {
        respond(404, ['error' => '任务不存在']);
    }
    $action = $m[2] ?? '';
    if ($action === '' && $method === 'GET') {
        $job['output_url'] = $job['state'] === 'done' ? "/api/jobs/$id/output" : null;
        respond(200, $job);
    }
    if ($action === '/retry' && $method === 'POST') {
        $update = query("UPDATE jobs SET state='queued', percent=0, stage='等待重试', error=NULL WHERE id=? AND state='failed'", [$id]);
        if (!$update->rowCount()) {
            respond(409, ['error' => '只能重试失败任务']);
        }
        respond(202, ['id' => $id, 'state' => 'queued']);
    }
    if ($action !== '/output' || $method !== 'GET') {
        respond(405, ['error' => '请求方法不支持']);
    }
    if ($job['state'] !== 'done') {
        respond(409, ['error' => '成片尚未就绪']);
    }
    $file = dataDir() . "/$id/result.mp4";
    if (!is_file($file)) {
        throw new RuntimeException('成片文件缺失');
    }
    $filename = pathinfo($job['name'], PATHINFO_FILENAME) . '_处理后.mp4';
    header('Content-Type: video/mp4');
    header("Content-Disposition: attachment; filename=\"processed.mp4\"; filename*=UTF-8''" . rawurlencode($filename));
    header('Content-Length: ' . filesize($file));
    readfile($file);
} catch (InvalidArgumentException $e) {
    respond(422, ['error' => $e->getMessage()]);
} catch (Throwable $e) {
    error_log((string)$e);
    respond(500, ['error' => '服务未能完成请求，请查看服务日志']);
}
