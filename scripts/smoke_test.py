"""Smoke test against the deployed API: health, a full session on a sample clip set, trace, delete.

  python scripts/smoke_test.py --stack unmask --region us-east-1
  python scripts/smoke_test.py --api https://xxxx.execute-api.us-east-1.amazonaws.com
  python scripts/smoke_test.py --api http://localhost:8000/api     # local server

Records Lambda (server-side) latencies to docs/results/lambda_latency.json.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def call(method, url, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"content-type": "application/json"} if data else {})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=40) as r:
        out = json.loads(r.read() or b"{}")
    return out, (time.perf_counter() - t0) * 1000


def post_form(url, fields, filename, data, ctype):
    b = uuid.uuid4().hex
    parts = []
    for k, v in fields.items():
        parts.append(f"--{b}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
    parts.append(f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n"
                 f"Content-Type: {ctype}\r\n\r\n".encode() + data + b"\r\n")
    parts.append(f"--{b}--\r\n".encode())
    req = urllib.request.Request(url, data=b"".join(parts), method="POST",
                                 headers={"content-type": f"multipart/form-data; boundary={b}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.status


def stack_api(stack, region):
    return subprocess.check_output(["aws", "cloudformation", "describe-stacks", "--stack-name", stack, "--region", region,
                                    "--query", "Stacks[0].Outputs[?OutputKey=='ApiUrl'].OutputValue", "--output", "text"],
                                   text=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api")
    ap.add_argument("--stack", default="unmask")
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--sample", default=None, help="sample id from frontend/samples/manifest.json")
    ap.add_argument("--no-record", action="store_true")
    a = ap.parse_args()
    api = (a.api or stack_api(a.stack, a.region)).rstrip("/")
    manifest = json.loads((ROOT / "frontend" / "samples" / "manifest.json").read_text())
    sample = next((s for s in manifest["samples"] if s["id"] == a.sample), manifest["samples"][0])
    print(f"API {api}\nsample {sample['id']}")

    h, ms = call("GET", f"{api}/health")
    assert h["ok"] and h["opencv"].startswith("5"), h
    print(f"health ok: OpenCV {h['opencv']}, config {h['config']}, bedrock={h['bedrock']} ({ms:.0f} ms)")

    s, _ = call("POST", f"{api}/sessions", {"mode": "upload", "available_challenges":
                                            [c for c in sample["clips"] if c != "passive"]})
    sid = s["session_id"]
    lat = []
    used = set()
    for _ in range(12):
        if s["status"] != "active":
            break
        na = s["next_action"]
        ch = na.get("challenge") or "passive"
        if na["action"] in ("record_passive", "issue_challenge") and ch in sample["clips"] and ch not in used:
            used.add(ch)
            data = (ROOT / "frontend" / sample["clips"][ch]).read_bytes()
            ctype = "video/webm" if sample["clips"][ch].endswith(".webm") else "video/mp4"
            u, _ = call("POST", f"{api}/sessions/{sid}/upload-url", {"challenge": ch, "content_type": ctype})
            st = post_form(u["upload"]["url"], u["upload"]["fields"], f"{ch}.mp4", data, ctype)
            assert st in (200, 201, 204), st
            s, ms = call("POST", f"{api}/sessions/{sid}/clips", {"key": u["key"], "challenge": ch})
            c = s["clip"]
            lat.append({"challenge": ch, "roundtrip_ms": round(ms, 1), "analysis_ms": c["timing_ms"]["total"],
                        "frames": c["timing_ms"].get("frames"), "usable": c["usable"]})
            stp = s["step"]
            print(f"  {ch:<12} usable={c['usable']!s:<5} {ms:6.0f} ms round trip, {c['timing_ms']['total']:6.0f} ms OpenCV"
                  f" -> {stp['chose']}({stp.get('challenge') or ''}) [{stp['policy']}] {stp['why']}")
        else:
            s, _ = call("POST", f"{api}/sessions/{sid}/skip", {"challenge": ch})
    if s["status"] == "active":
        s, _ = call("POST", f"{api}/sessions/{sid}/finish")
    print(f"verdict: {s['result']['verdict']} (risk {s['result']['risk']})")
    t, _ = call("GET", f"{api}/sessions/{sid}/trace")
    assert t["trace"] and all("decision" in x for x in t["trace"])
    print(f"trace: {len(t['trace'])} steps, policies {[x['decision'].get('policy') for x in t['trace']]}")
    d, _ = call("DELETE", f"{api}/sessions/{sid}")
    assert d.get("deleted") == sid
    print("deleted session data")
    if lat and not a.no_record and "localhost" not in api:
        out = {"api": api.split("//")[1].split(".")[0] + ".execute-api", "measured_at": time.strftime("%Y-%m-%d %H:%M"),
               "clips": lat,
               "roundtrip_ms_median": round(statistics.median(x["roundtrip_ms"] for x in lat), 1),
               "analysis_ms_median": round(statistics.median(x["analysis_ms"] for x in lat), 1),
               "note": "roundtrip = client->API Gateway->Lambda (analysis + agent + DynamoDB) -> client; includes cold start on first clip"}
        p = ROOT / "docs" / "results" / "lambda_latency.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2))
        print(f"wrote {p}")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        sys.exit(f"SMOKE TEST FAILED: {e}")
