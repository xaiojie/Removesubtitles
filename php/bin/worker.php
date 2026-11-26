<?php
declare(strict_types=1);
require dirname(__DIR__) . '/common.php';

// Initialize the same schema/migrations as the API before handing its PID to the supervisor.
db();
$command = ['/usr/bin/env', 'python3', __DIR__ . '/scheduler.py', ...array_slice($argv, 1)];
if (function_exists('pcntl_exec')) {
    pcntl_exec($command[0], array_slice($command, 1));
    throw new RuntimeException('无法启动队列调度器，请检查 Python 3');
}
$process = proc_open($command, [STDIN, STDOUT, STDERR], $pipes);
exit(is_resource($process) ? proc_close($process) : 1);
