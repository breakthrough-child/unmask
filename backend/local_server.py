"""Run the whole app locally (no AWS): static frontend + the Lambda handler.

  python -m backend.local_server --port 8000
  open http://localhost:8000

Uses LocalStorage (./.local-store) and, unless --bedrock is given, the
rule-based policy. The handler code is exactly the one deployed to Lambda.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import sys
from email.parser import BytesParser
from email.policy import default as email_policy
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROUTES = [
    ("GET", r"^/health$", "/health"),
    ("POST", r"^/sessions$", "/sessions"),
    ("GET", r"^/sessions/(?P<id>[^/]+)$", "/sessions/{id}"),
    ("DELETE", r"^/sessions/(?P<id>[^/]+)$", "/sessions/{id}"),
    ("GET", r"^/sessions/(?P<id>[^/]+)/trace$", "/sessions/{id}/trace"),
    ("POST", r"^/sessions/(?P<id>[^/]+)/upload-url$", "/sessions/{id}/upload-url"),
    ("POST", r"^/sessions/(?P<id>[^/]+)/clips$", "/sessions/{id}/clips"),
    ("POST", r"^/sessions/(?P<id>[^/]+)/skip$", "/sessions/{id}/skip"),
    ("POST", r"^/sessions/(?P<id>[^/]+)/finish$", "/sessions/{id}/finish"),
]


class Ctx:
    aws_request_id = "local"


def make_handler(app, store, static_dir):
    class H(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(static_dir), **kw)

        def log_message(self, fmt, *args):
            sys.stderr.write("[http] " + fmt % args + "\n")

        def _api(self, method):
            path = self.path.split("?")[0]
            api_path = path[4:] if path.startswith("/api/") else None
            if api_path is None:
                return False
            for m, rx, key in ROUTES:
                mt = re.match(rx, api_path)
                if m == method and mt:
                    n = int(self.headers.get("content-length") or 0)
                    body = self.rfile.read(n).decode() if n else None
                    ev = {"routeKey": f"{method} {key}", "pathParameters": mt.groupdict() or None, "body": body,
                          "requestContext": {"http": {"method": method}}}
                    r = app.handler(ev, Ctx())
                    self._send(r["statusCode"], r["body"].encode(), "application/json")
                    return True
            self._send(404, b'{"error":"no such route"}', "application/json")
            return True

        def _send(self, code, data, ctype):
            self.send_response(code)
            self.send_header("content-type", ctype)
            self.send_header("content-length", str(len(data)))
            self.send_header("cache-control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self._api("GET"):
                return
            if self.path.startswith("/_files/"):
                try:
                    p = store.path(self.path[len("/_files/"):].split("?")[0])
                    self._send(200, p.read_bytes(), mimetypes.guess_type(str(p))[0] or "application/octet-stream")
                except (ValueError, FileNotFoundError):
                    self._send(404, b"not found", "text/plain")
                return
            if self.path.split("?")[0] == "/config.js":
                self._send(200, b'window.UNMASK_CONFIG = {apiUrl: "/api", mode: "local"};', "application/javascript")
                return
            super().do_GET()

        def do_DELETE(self):
            self._api("DELETE")

        def do_POST(self):
            if self._api("POST"):
                return
            if self.path == "/_upload":
                n = int(self.headers.get("content-length") or 0)
                raw = self.rfile.read(n)
                msg = BytesParser(policy=email_policy).parsebytes(
                    b"content-type: " + self.headers["content-type"].encode() + b"\r\n\r\n" + raw)
                fields, data = {}, None
                for part in msg.iter_parts():
                    name = part.get_param("name", header="content-disposition")
                    if name == "file":
                        data = part.get_payload(decode=True)
                    else:
                        fields[name] = part.get_content().strip()
                key = fields.get("key", "")
                if not key.startswith("clips/") or data is None:
                    self._send(400, b"bad upload", "text/plain")
                    return
                store.put_bytes(key, data, fields.get("Content-Type", "video/webm"))
                self._send(204, b"", "text/plain")
                return
            self._send(404, b"not found", "text/plain")

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--bedrock", action="store_true", help="call Bedrock (needs AWS credentials)")
    ap.add_argument("--store", default=str(ROOT / ".local-store"))
    a = ap.parse_args()
    os.environ["UNMASK_STORAGE"] = "local"
    os.environ["UNMASK_LOCAL_ROOT"] = a.store
    os.environ["UNMASK_LOCAL_URL"] = f"http://localhost:{a.port}"
    os.environ["USE_BEDROCK"] = "true" if a.bedrock else "false"
    sys.path.insert(0, str(ROOT))
    from backend import app
    app.store.purge_older_than()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(app, app.store, ROOT / "frontend"))
    print(json.dumps({"event": "local_server", "url": f"http://localhost:{a.port}", "bedrock": a.bedrock}))
    srv.serve_forever()


if __name__ == "__main__":
    main()
