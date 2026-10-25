"""Observability: structured JSON logs + CloudWatch metrics via Embedded Metric Format.

EMF lines are parsed by CloudWatch Logs into custom metrics (namespace Unmask),
so no PutMetricData calls or extra IAM permissions are needed."""
from __future__ import annotations

import json
import os
import sys
import time

NAMESPACE = os.environ.get("METRICS_NAMESPACE", "Unmask")
_ctx: dict = {}


def set_context(**kw):
    _ctx.clear()
    _ctx.update({k: v for k, v in kw.items() if v is not None})


def log(event: str, level: str = "INFO", **fields):
    rec = {"ts": round(time.time(), 3), "level": level, "event": event, **_ctx, **fields}
    print(json.dumps(rec, default=str), file=sys.stdout, flush=True)


def metric(name: str, value: float, unit: str = "Count", **dims):
    dims = {k: str(v) for k, v in dims.items()}
    rec = {
        "_aws": {"Timestamp": int(time.time() * 1000), "CloudWatchMetrics": [{
            "Namespace": NAMESPACE,
            "Dimensions": [list(dims)] if dims else [[]],
            "Metrics": [{"Name": name, "Unit": unit}]}]},
        name: value, **dims,
    }
    print(json.dumps(rec), file=sys.stdout, flush=True)
