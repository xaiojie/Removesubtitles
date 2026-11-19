<?php
declare(strict_types=1);
require dirname(__DIR__) . '/common.php';

function processJob(string $id, string $mode, string $folder, $lock): void
{
    $python = $mode === 'lite'
        ? (getenv('VSR_PYTHON') ?: ROOT . '/video-subtitle-remover/.venv/bin/python')
        : (getenv('CLEAR_PYTHON') ?: ROOT . '/.venv/bin/python');
    if (!is_executable($python)) {
        throw new RuntimeException('推理 Python 不可用，请检查对应模型环境');
    }
    $log = fopen("$folder/worker.log", 'ab');
    if (!$log) {
        throw new RuntimeException('无法打开任务日志');
    }
    try {
        // Keep the lock in the child as well, in case the PHP worker is restarted mid-job.
        $descriptors = [0 => ['file', '/dev/null', 'r'], 1 => ['pipe', 'w'], 2 => ['redirect', 1], 3 => $lock];
        $command = [$python, '-u', dirname(__DIR__) . '/inference/run.py', $folder, $mode];
        $proc = proc_open($command, $descriptors, $pipes);
        if (!is_resource($proc)) {
            throw new RuntimeException('无法启动推理');
        }
        $reason = '推理失败，请检查模型和任务日志';
        try {
            while (($line = fgets($pipes[1])) !== false) {
                fwrite($log, $line);
                if (!str_starts_with($line, 'JOB_STATUS ')) {
                    continue;
                }
                $event = json_decode(substr($line, strlen('JOB_STATUS ')), true);
                if (isset($event['error'])) {
                    $reason = (string)$event['error'];
                }
                if (isset($event['percent'], $event['stage'])) {
                    $percent = min(99, max(0, (int)$event['percent']));
                    query('UPDATE jobs SET percent=max(percent, CAST(? AS INTEGER)), stage=? WHERE id=?',
                        [$percent, (string)$event['stage'], $id]);
                }
            }
        } finally {
            fclose($pipes[1]);
            $code = proc_close($proc);
        }
        if ($code !== 0) {
            throw new RuntimeException($reason);
        }
        if (!is_file("$folder/result.mp4") || !filesize("$folder/result.mp4")) {
            throw new RuntimeException('推理未生成成片');
        }
    } finally {
        fclose($log);
    }
}

$dir = dataDir();
$once = in_array('--once', $argv, true);
$lock = fopen("$dir/worker.lock", 'c');
if (!$lock) {
    throw new RuntimeException('无法打开队列锁');
}
if (!flock($lock, $once ? LOCK_EX | LOCK_NB : LOCK_EX)) {
    fwrite(STDERR, "队列正在处理其他任务\n");
    exit(1);
}
query("UPDATE jobs SET state='failed', stage='任务中断', error='队列进程曾退出，可提交重试' WHERE state='running'");
do {
    $job = query("SELECT id, mode FROM jobs WHERE state='queued' ORDER BY created, rowid LIMIT 1")->fetch(PDO::FETCH_ASSOC);
    if (!$job) {
        if ($once) {
            break;
        }
        sleep(1);
        continue;
    }
    $id = $job['id'];
    query("UPDATE jobs SET state='running', stage='准备推理', error=NULL WHERE id=?", [$id]);
    try {
        processJob($id, $job['mode'], "$dir/$id", $lock);
        query("UPDATE jobs SET state='done', percent=100, stage='处理完成' WHERE id=?", [$id]);
    } catch (Throwable $e) {
        query("UPDATE jobs SET state='failed', stage='处理失败', error=? WHERE id=?", [$e->getMessage(), $id]);
    }
} while (!$once);
