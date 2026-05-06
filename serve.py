#!/usr/bin/env python3
"""Multi-threaded HTTP server with COOP/COEP headers for SharedArrayBuffer (ONNX Runtime Web WASM)"""
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import os

PORT = 8080
DIR = os.path.dirname(os.path.abspath(__file__))

class CORSHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIR, **kwargs)

    def end_headers(self):
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

    def log_message(self, format, *args):
        pass

if __name__ == '__main__':
    server = ThreadingHTTPServer(('0.0.0.0', PORT), CORSHandler)
    print('HTTP Server with COOP/COEP headers on port', PORT)
    print('Open: http://localhost:{}/古建监测大屏_v5.html'.format(PORT))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
