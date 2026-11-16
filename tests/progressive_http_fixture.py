"""Isolated HTTP fixture for Blender UI tests; no fixtures are saved to disk."""
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import time
import threading

configuration = json.loads(sys.stdin.readline())
payloads = {name: base64.b64decode(value) for name, value in configuration['payloads'].items()}
guard = threading.Lock()
active = peak = 0
requests = []
request_headers = []


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        global active, peak
        if self.path == '/stats':
            with guard:
                content = json.dumps({'peak': peak, 'requests': requests,
                                      'request_headers': request_headers}).encode()
            self.send_response(200)
            self.send_header('Content-Length', str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return
        with guard:
            active += 1
            peak = max(peak, active)
            requests.append(self.path)
            request_headers.append({name: self.headers.get(name)
                                    for name in configuration.get('record_headers', ())})
        try:
            status, content, length = self.prepare_payload()
        except BaseException:
            with guard:
                active -= 1
            raise
        # Retire the request before another handler can count itself after the
        # client has received this response. Delays remain outside the lock, so
        # genuinely overlapping downloads still contribute to the peak.
        with guard:
            try:
                self.send_response(status)
                self.send_header('Content-Length', str(length))
                self.end_headers()
                self.wfile.write(content)
            except ConnectionError:
                pass
            finally:
                active -= 1

    def prepare_payload(self):
        key = self.path.split('?', 1)[0].lstrip('/')
        time.sleep(configuration.get('delays', {}).get(key, 0))
        # Optional UI-test barrier: do not race a fixed sleep against viewport
        # redraw speed. Other transport tests omit gates and retain their timing.
        gate = configuration.get('gates', {}).get(key)
        if gate:
            deadline = time.monotonic() + 120
            while not Path(gate).exists():
                if time.monotonic() > deadline:
                    content = b'Test gate timed out'
                    return 503, content, len(content)
                time.sleep(0.02)
        content = payloads.get(key)
        if content is None:
            content = b'Not found'
            return 404, content, len(content)
        length = len(content)
        with guard:
            first_attempt = requests.count(self.path) == 1
        if key in configuration.get('truncate_once', ()) and first_attempt:
            content = content[:max(1, len(content) // 2)]
        return 200, content, length


server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
print(server.server_port, flush=True)
server.serve_forever()
