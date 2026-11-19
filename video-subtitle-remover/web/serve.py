"""双端口启动器：同一个 Web 应用同时监听 8800 和 8801。

外网 20876 / 20877 分别映射到内网 8800 / 8801，因此两个域名/端口
访问到的是完全相同的应用，任务队列、进度、历史记录共享。
"""
import asyncio
import os
import signal
import sys

import uvicorn

APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)

from app import app  # noqa: E402

PORTS = [int(p) for p in os.environ.get('VSR_PORTS', '8800,8801').split(',') if p.strip()]
HOST = os.environ.get('VSR_HOST', '0.0.0.0')


async def main():
    servers = []
    for port in PORTS:
        cfg = uvicorn.Config(app, host=HOST, port=port, log_level='info',
                             access_log=False, lifespan='off')
        server = uvicorn.Server(cfg)
        server.install_signal_handlers = lambda: None  # 统一在下面处理
        servers.append(server)

    loop = asyncio.get_running_loop()

    def _shutdown(*_args):
        for s in servers:
            s.should_exit = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _shutdown)
        except NotImplementedError:
            pass

    print(f'[vsr-web] 监听 {HOST}:{", ".join(str(p) for p in PORTS)}', flush=True)
    await asyncio.gather(*(s.serve() for s in servers))


if __name__ == '__main__':
    asyncio.run(main())
