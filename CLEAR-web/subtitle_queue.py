"""Web's server-side client for the canonical subtitle queue."""

from http.client import HTTPConnection, HTTPSConnection, HTTPException
import json
import os
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4


class QueueError(Exception):
    def __init__(self, message, status=503):
        super().__init__(message)
        self.status = status


class SubtitleQueue:
    def __init__(self, workspace):
        directory = workspace / '.deploy/api'
        try:
            settings = json.loads((directory / 'settings.json').read_text(encoding='utf-8'))
        except FileNotFoundError:
            settings = {}
        except (OSError, ValueError) as error:
            raise QueueError('无法读取字幕队列部署配置。') from error
        host = settings.get('bind', '127.0.0.1')
        host = {'0.0.0.0': '127.0.0.1', '::': '::1'}.get(host, host)
        if ':' in host:
            host = f'[{host}]'
        default_url = f"http://{host}:{settings.get('port', 8810)}"
        self.base = urlsplit(os.environ.get('SUBTITLE_API_URL') or default_url)
        self.data = Path(os.environ.get('SUBTITLE_DATA_DIR') or settings.get('data') or workspace / 'php/var')
        self.token_file = Path(os.environ.get('SUBTITLE_API_TOKEN_FILE') or directory / 'api.token')

    def json_request(self, method, path, *, body=None, headers=None):
        token = os.environ.get('SUBTITLE_API_TOKEN')
        if not token:
            try:
                token = self.token_file.read_text(encoding='utf-8').strip()
            except FileNotFoundError:
                token = ''
            except OSError as error:
                raise QueueError('网页服务无法读取字幕队列的接口令牌。') from error
        fields = {'Accept': 'application/json', 'X-Subtitle-Client': 'web', **(headers or {})}
        if token:
            fields['Authorization'] = 'Bearer ' + token
        if self.base.scheme not in ('http', 'https') or not self.base.hostname:
            raise QueueError('字幕队列接口地址无效。')
        connection = HTTPSConnection if self.base.scheme == 'https' else HTTPConnection
        try:
            server = connection(self.base.hostname, self.base.port, timeout=90 if body is not None else 5, blocksize=1024 * 1024)
            try:
                server.request(method, self.base.path.rstrip('/') + path, body=body, headers=fields)
                response = server.getresponse()
                data = json.load(response)
                status = response.status
            finally:
                server.close()
        except (OSError, ValueError, HTTPException) as error:
            raise QueueError('字幕队列暂时无法连接，请检查 PHP 服务地址和运行状态。') from error
        if status in (401, 403):
            raise QueueError('网页服务连接字幕队列的身份验证失败。')
        if not (200 <= status < 300) or not isinstance(data, dict):
            message = data.get('error') if isinstance(data, dict) else None
            raise QueueError(message or '字幕队列返回了无效响应。', status if 400 <= status < 600 else 502)
        return data, status

    def submit(self, form, video):
        # Flask has already parsed the upload; stream its spool, including large files, to PHP.
        boundary = uuid4().hex
        prefix = bytearray()
        for key in ('mode', 'regions', 'dimensions', 'catalog_sha256', 'engine', 'inference'):
            if key in form:
                prefix.extend((f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n'
                               f'{form[key]}\r\n').encode('utf-8'))
        name = video.filename.replace('\\', '/').rsplit('/', 1)[-1]
        name = name.replace('"', '%22').replace('\r', '%0D').replace('\n', '%0A')
        prefix.extend((f'--{boundary}\r\nContent-Disposition: form-data; name="video"; filename="{name}"\r\n'
                       'Content-Type: application/octet-stream\r\n\r\n').encode('utf-8'))
        suffix = f'\r\n--{boundary}--\r\n'.encode('ascii')
        size = video.stream.seek(0, os.SEEK_END)
        video.stream.seek(0)

        def multipart():
            yield prefix
            while chunk := video.stream.read(1024 * 1024):
                yield chunk
            yield suffix

        return self.json_request('POST', '/api/jobs', body=multipart(), headers={
            'Content-Type': 'multipart/form-data; boundary=' + boundary,
            'Content-Length': str(len(prefix) + size + len(suffix)),
        })
