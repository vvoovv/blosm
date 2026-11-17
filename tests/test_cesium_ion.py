"""Endpoint resolution contracts, including the real shared HTTP transport."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib
import json
from pathlib import Path
import sys
import threading
import time
import types
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
from release_target import ADDON_ROOT

# Exercise the Blender-independent resolver without executing the addon entry point.
package = types.ModuleType('_blosm_ion_tests')
package.__path__ = [str(ADDON_ROOT)]
sys.modules[package.__name__] = package
ion = importlib.import_module(package.__name__ + '.threed_tiles.cesium_ion')
Job = importlib.import_module(package.__name__ + '.progressive.core').Job
DownloadTransport = importlib.import_module(package.__name__ + '.downloads.transport').DownloadTransport

TOKEN = 'test-ion-credential'
GOOGLE_URL = 'https://tile.googleapis.com/v1/3dtiles/root.json?key=test-provider-credential'


def external():
    return {'type': '3DTILES', 'externalType': '3DTILES', 'options': {'url': GOOGLE_URL},
            'attributions': [{'html': '<span>Cesium ion / Google Maps ©</span>', 'collapsible': False}]}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        self.server.requests.append((self.path, self.headers.get('Authorization')))
        payload = self.server.payload
        self.send_response(self.server.status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class ResolutionChecks(unittest.TestCase):
    def test_endpoint_response_size_is_bounded(self):
        with self.assertRaisesRegex(ion.AssetResolutionError, '1 MiB'):
            ion.parse_endpoint(b' ' * (1024 * 1024 + 1))

    def test_request_uses_ion_bearer_header_and_verified_tls(self):
        request = ion.endpoint_request('  ' + TOKEN + '\n')
        self.assertEqual(request.url, 'https://api.cesium.com/v1/assets/2275207/endpoint')
        self.assertEqual(request.headers['Authorization'], 'Bearer ' + TOKEN)
        self.assertTrue(request.verify_tls)
        self.assertNotIn(TOKEN, request.url)
        for token in ('', '  ', 'bad\r\ntoken', 'bad token'):
            with self.subTest(token=repr(token)), self.assertRaises(ion.AssetResolutionError):
                ion.endpoint_request(token)

    def test_external_google_response_preserves_url_and_attribution(self):
        result = ion.parse_endpoint(json.dumps(external()).encode())
        self.assertEqual(result.asset_id, 2275207)
        self.assertEqual(result.url, GOOGLE_URL)
        self.assertEqual(result.headers, {})
        self.assertEqual(result.attributions[0].html, '<span>Cesium ion / Google Maps ©</span>')
        self.assertFalse(result.attributions[0].collapsible)
        self.assertNotIn('test-provider-credential', repr(result))

    def test_hosted_response_uses_returned_token(self):
        result = ion.parse_endpoint(json.dumps({'type': '3DTILES',
            'url': 'https://assets.ion.cesium.com/asset/tileset.json', 'accessToken': 'endpoint-credential'}))
        self.assertEqual(result.headers, {'Authorization': 'Bearer endpoint-credential'})
        self.assertNotIn('endpoint-credential', repr(result))

    def test_rejects_malformed_or_unsupported_responses_without_echoing_them(self):
        documents = [[], None, {}, {'type': 'TERRAIN'},
                     {'type': '3DTILES', 'externalType': 'OTHER'},
                     {'type': '3DTILES', 'url': GOOGLE_URL},
                     {'type': '3DTILES', 'externalType': '3DTILES', 'options': []}]
        for url in ('', 'http://host/root.json', 'file:///root.json',
                    'https://user:password@host/root.json', 'https://host/root.json#fragment',
                    'https://host:bad/root.json', 'https://host/root.json\n'):
            data = external()
            data['options']['url'] = url
            documents.append(data)
        for credit in ({}, ['not an object'], [{'html': 1}], [{'html': 'x', 'collapsible': 'no'}]):
            data = external()
            data['attributions'] = credit
            documents.append(data)
        for document in documents:
            with self.subTest(document=document), self.assertRaises(ion.AssetResolutionError):
                ion.parse_endpoint(json.dumps(document))
        for raw in (b'not JSON test-provider-credential', b'\xff', b'[' * 2000):
            with self.assertRaises(ion.AssetResolutionError) as caught:
                ion.parse_endpoint(raw)
            self.assertNotIn('test-provider-credential', str(caught.exception))

    def test_rejected_request_redacts_echoed_ion_token(self):
        steps = ion.resolve_asset(TOKEN)
        next(steps)
        with self.assertRaises(ion.AssetResolutionError) as caught:
            steps.throw(RuntimeError('Denied: ' + TOKEN))
        self.assertNotIn(TOKEN, str(caught.exception))

    def test_shared_transport_resolves_once_without_fetching_root_or_tiles(self):
        for status, payload in ((200, external()), (401, {'message': 'Denied: ' + TOKEN})):
            with self.subTest(status=status):
                server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
                server.requests, server.status, server.payload = [], status, json.dumps(payload).encode()
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                try:
                    api = 'http://127.0.0.1:%s/v1/assets/{asset_id}/endpoint' % server.server_port
                    with patch.object(ion, 'API_URL', api):
                        job = Job(ion.resolve_asset(TOKEN), DownloadTransport(max_requests=1, max_bytes=1024 * 1024))
                        deadline = time.monotonic() + 15
                        try:
                            while not job.done and time.monotonic() < deadline:
                                job.step()
                                time.sleep(.002)
                            self.assertTrue(job.done, 'Resolution timed out')
                            self.assertEqual(server.requests, [('/v1/assets/2275207/endpoint', 'Bearer ' + TOKEN)], str(job.error))
                            if status == 200:
                                self.assertIsNone(job.error)
                                self.assertEqual(job.result.url, GOOGLE_URL)
                            else:
                                self.assertIsInstance(job.error, ion.AssetResolutionError)
                                self.assertNotIn(TOKEN, str(job.error))
                        finally:
                            job.cancel()
                finally:
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=2)


if __name__ == '__main__':
    unittest.main()
