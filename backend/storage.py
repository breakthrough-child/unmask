"""Storage backends: AWS (S3 + DynamoDB) and Local (files, for dev / judges without AWS)."""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path


class Conflict(Exception):
    pass


class AwsStorage:
    def __init__(self, bucket: str, table: str):
        import boto3
        from botocore.config import Config
        self.bucket = bucket
        self.s3 = boto3.client("s3", config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"}))
        self.table = boto3.resource("dynamodb").Table(table)

    def get_session(self, sid):
        item = self.table.get_item(Key={"session_id": sid}).get("Item")
        if not item:
            return None
        s = json.loads(item["doc"])
        s["_version"] = int(item.get("version", 0))
        return s

    def put_session(self, s):
        ver = s.pop("_version", 0)
        try:
            self.table.put_item(
                Item={"session_id": s["session_id"], "doc": json.dumps(s, default=str), "version": ver + 1,
                      "expires_at": s["expires_at"], "status": s["status"],
                      "verdict": (s.get("result") or {}).get("verdict", "")},
                ConditionExpression="attribute_not_exists(session_id) OR version = :v",
                ExpressionAttributeValues={":v": ver})
        except self.table.meta.client.exceptions.ConditionalCheckFailedException:
            raise Conflict()
        s["_version"] = ver + 1

    def delete_session(self, sid):
        self.table.delete_item(Key={"session_id": sid})

    def presign_post(self, key, ctype, max_bytes):
        return self.s3.generate_presigned_post(
            self.bucket, key, Fields={"Content-Type": ctype},
            Conditions=[{"Content-Type": ctype}, ["content-length-range", 1, max_bytes]], ExpiresIn=600)

    def presign_get(self, key):
        return self.s3.generate_presigned_url("get_object", Params={"Bucket": self.bucket, "Key": key}, ExpiresIn=3600)

    def download(self, key, local):
        self.s3.download_file(self.bucket, key, local)

    def put_bytes(self, key, data, ctype):
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType=ctype)

    def delete_prefix(self, prefix):
        for page in self.s3.get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=prefix):
            objs = [{"Key": o["Key"]} for o in page.get("Contents", [])]
            if objs:
                self.s3.delete_objects(Bucket=self.bucket, Delete={"Objects": objs})


class LocalStorage:
    """Mirrors AwsStorage on the local filesystem. URLs point at backend/local_server.py."""

    def __init__(self, root: str, base_url: str):
        self.root = Path(root)
        self.base = base_url.rstrip("/")
        (self.root / "sessions").mkdir(parents=True, exist_ok=True)

    def _sp(self, sid):
        return self.root / "sessions" / f"{sid}.json"

    def get_session(self, sid):
        p = self._sp(sid)
        if not p.exists():
            return None
        s = json.loads(p.read_text())
        s.setdefault("_version", 0)
        return s

    def put_session(self, s):
        cur = self.get_session(s["session_id"])
        ver = s.pop("_version", 0)
        if cur is not None and cur.get("_version", 0) != ver:
            raise Conflict()
        s["_version"] = ver + 1
        self._sp(s["session_id"]).write_text(json.dumps(s, default=str))

    def delete_session(self, sid):
        self._sp(sid).unlink(missing_ok=True)

    def presign_post(self, key, ctype, max_bytes):
        return {"url": f"{self.base}/_upload", "fields": {"key": key, "Content-Type": ctype}}

    def presign_get(self, key):
        return f"{self.base}/_files/{key}"

    def path(self, key) -> Path:
        p = (self.root / key).resolve()
        if self.root.resolve() not in p.parents:
            raise ValueError("bad key")
        return p

    def download(self, key, local):
        shutil.copyfile(self.path(key), local)

    def put_bytes(self, key, data, ctype):
        p = self.path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def delete_prefix(self, prefix):
        p = self.root / prefix
        if p.exists():
            shutil.rmtree(p)

    def purge_older_than(self, seconds=24 * 3600):
        """Same 24h retention as the S3 lifecycle rule."""
        cutoff = time.time() - seconds
        for f in self.root.rglob("*"):
            if f.is_file() and f.stat().st_mtime < cutoff:
                f.unlink()


def from_env():
    if os.environ.get("UNMASK_STORAGE") == "local":
        return LocalStorage(os.environ.get("UNMASK_LOCAL_ROOT", ".local-store"),
                            os.environ.get("UNMASK_LOCAL_URL", "http://localhost:8000"))
    return AwsStorage(os.environ["CLIP_BUCKET"], os.environ["SESSION_TABLE"])
