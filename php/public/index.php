<?php
declare(strict_types=1);
require dirname(__DIR__) . '/common.php';

function respond(int $code, array $value): never
{
    http_response_code($code);
    header('Content-Type: application/json; charset=utf-8');
    echo json_encode($value, JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES | JSON_PRESERVE_ZERO_FRACTION | JSON_THROW_ON_ERROR);
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
    $catalog = productCatalog();
    $mode = $_POST['mode'] ?? $catalog['default'];
    if (!is_string($mode) || !isset($catalog['modes'][$mode])) {
        respond(422, ['error' => 'mode 可选 ' . implode('、', array_keys($catalog['modes']))]);
    }
    if (array_key_exists('engine', $_POST) || array_key_exists('inference', $_POST)) {
        respond(422, ['error' => '三档处理使用所选档位的固定参数']);
    }
    if (isset($_POST['catalog_sha256']) && $_POST['catalog_sha256'] !== $catalog['catalog_sha256']) {
        respond(409, ['error' => '处理档位配置已更新，请刷新后重新提交']);
    }
    $raw = $_POST['regions'] ?? '';
    if (!is_string($raw)) {
        respond(422, ['error' => 'regions 需要 JSON 数组']);
    }
    $size = dimensions($tmp);
    if (isset($_POST['dimensions']) && (!is_string($_POST['dimensions'])
        || json_decode($_POST['dimensions'], true) !== $size)) {
        respond(422, ['error' => '选区使用的显示尺寸与视频不一致']);
    }
    $regions = validRegions(json_decode($raw), $size);
    $plan = resolveProductPlan($mode, array_column($regions, 'type'));
    $source = ($_SERVER['HTTP_X_SUBTITLE_CLIENT'] ?? '') === 'web' ? 'web' : 'api';
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
        $config = ['mode' => $mode, 'dimensions' => $size, 'regions' => $regions, 'original_filename' => $name,
            'submitted_via' => $source, 'resolved_plan' => $plan];
        $json = json_encode($config, JSON_THROW_ON_ERROR | JSON_PRESERVE_ZERO_FRACTION | JSON_UNESCAPED_UNICODE);
        if (file_put_contents("$dir/request.json", $json) === false) {
            throw new RuntimeException('保存任务信息失败');
        }
        $now = microtime(true);
        query("INSERT INTO jobs (id, name, mode, state, stage, created, queued_at) VALUES (?, ?, ?, 'queued', '排队中', ?, ?)",
            [$id, $name, $mode, (int)$now, $now]);
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
        'submitted_via' => $source, 'resolved_plan' => $plan,
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
        respond(200, productCatalog());
    }
    if ($method === 'POST' && ($path === '/api/jobs' || str_ends_with($path, '/retry'))) {
        // Hold until submission commits. Maintenance takes the exclusive lock before draining.
        $submission = fopen(dataDir() . '/submissions.lock', 'c');
        if (!$submission) {
            throw new RuntimeException('无法打开队列提交锁');
        }
        if (!flock($submission, LOCK_SH | LOCK_NB)) {
            respond(503, ['error' => '服务正在维护，请稍后重新提交']);
        }
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
        $folder = dataDir() . "/$id";
        $config = is_file("$folder/request.json") ? json_decode(file_get_contents("$folder/request.json"), true) : [];
        $job['output_url'] = $job['state'] === 'done' ? "/api/jobs/$id/output" : null;
        $job['can_retry'] = $job['state'] === 'failed';
        $job['submitted_via'] = $config['submitted_via'] ?? null;
        $job['resolved_plan'] = $config['resolved_plan'] ?? null;
        $job['regions'] = $config['regions'] ?? null;
        $job['dimensions'] = $config['dimensions'] ?? null;
        $job['pipeline'] = $job['resolved_plan']['stages'] ?? null;
        if ($job['state'] === 'done' && is_file("$folder/standard/quality.json")) {
            $job['quality'] = json_decode(file_get_contents("$folder/standard/quality.json"), true);
        }
        $queued = (float)($job['queued_at'] ?? $job['created']);
        if ($job['state'] === 'queued') {
            $job['queue_wait_seconds'] = round(max(0, microtime(true) - $queued), 2);
        } elseif (query("SELECT 1 FROM sqlite_master WHERE type='table' AND name='job_runs'")->fetchColumn()) {
            $run = query('SELECT run_id, started_at, finished_at FROM job_runs WHERE job_id=? ORDER BY started_at DESC LIMIT 1',
                [$id])->fetch(PDO::FETCH_ASSOC);
            if ($run) {
                $job['run_id'] = $run['run_id'];
                $job['started_at'] = (float)$run['started_at'];
                $job['queue_wait_seconds'] = round(max(0, $job['started_at'] - $queued), 2);
                $job['elapsed_seconds'] = round(max(0, ($run['finished_at'] ?? microtime(true)) - $job['started_at']), 2);
            }
        }
        if ($job['state'] !== 'queued' && is_file("$folder/execution.json")) {
            $execution = json_decode(file_get_contents("$folder/execution.json"), true);
            if (($execution['run_id'] ?? null) === ($job['run_id'] ?? null)) {
                $job['execution'] = $execution;
            }
        }
        respond(200, $job);
    }
    if ($action === '/retry' && $method === 'POST') {
        $update = query("UPDATE jobs SET state='queued', queued_at=?, percent=0, stage='等待重试', error=NULL WHERE id=? AND state='failed'", [microtime(true), $id]);
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
