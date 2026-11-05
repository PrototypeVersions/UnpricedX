"""Publish a manifest and verify the actual HTTP content, without market-data dependencies."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

MANIFEST = "data/deployment.json"
REQUIRED = ("index.html", "app.js", "style.css", "data/status.json", "data/forecasts.json",
            "data/live.json", "data/export/latest.json")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def prepare(site: Path, commit: str) -> dict:
    for name in REQUIRED:
        if not (site / name).is_file():
            raise ValueError(f"required site file missing: {name}")
    files = {}
    for path in sorted(site.rglob("*")):
        if not path.is_file():
            continue
        name = path.relative_to(site).as_posix()
        if name == MANIFEST:
            continue
        data = path.read_bytes()
        if path.suffix == ".json":
            # Parsing alone permits NaN; serializing with allow_nan=False rejects it.
            json.dumps(json.loads(data), allow_nan=False)
        files[name] = digest(data)
    status = json.loads((site / "data/status.json").read_text())
    manifest = {"schema": "unpriced-deployment/1", "sourceCommit": commit,
                "builtAt": datetime.now(timezone.utc).isoformat(),
                "forecastSession": status.get("lastSession"), "files": files}
    (site / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {len(files)} files from {commit}; forecast session {manifest['forecastSession']}")
    return manifest


def fetch(url: str) -> bytes:
    req = Request(url, headers={"Cache-Control": "no-cache", "User-Agent": "UnpricedX-deployment-check"})
    with urlopen(req, timeout=20) as response:
        return response.read()


def verify(site: Path, base_url: str, fetcher=fetch) -> dict:
    if not base_url.startswith("https://"):
        raise ValueError("a public HTTPS Pages URL is required")
    expected = json.loads((site / MANIFEST).read_text())
    # A unique query bypasses old edge-cache entries after GitHub reports success.
    suffix = f"?deployment={quote(expected['sourceCommit'])}&check={time.time_ns()}"
    base_url = base_url.rstrip("/") + "/"
    actual = json.loads(fetcher(base_url + MANIFEST + suffix))
    if actual != expected:
        raise ValueError("public deployment manifest is stale or differs from the uploaded artifact")
    for name, sha in expected["files"].items():
        if digest(fetcher(base_url + quote(name, safe="/") + suffix)) != sha:
            raise ValueError(f"public file does not match the deployment: {name}")
    return expected


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["prepare", "verify"])
    p.add_argument("--site", type=Path, default=Path("site"))
    p.add_argument("--commit")
    p.add_argument("--url")
    p.add_argument("--attempts", type=int, default=1)
    args = p.parse_args()
    if args.command == "prepare":
        if not args.commit:
            p.error("prepare requires --commit")
        prepare(args.site, args.commit)
        return
    if not args.url or args.attempts < 1:
        p.error("verify requires --url and a positive number of attempts")
    for attempt in range(args.attempts):
        try:
            manifest = verify(args.site, args.url)
            print(f"Verified all {len(manifest['files'])} public files from {manifest['sourceCommit']}; "
                  f"forecast session {manifest['forecastSession']}")
            return
        except Exception as exc:
            if attempt == args.attempts - 1:
                raise
            print(f"Publication check {attempt + 1}/{args.attempts}: {exc}", flush=True)
            time.sleep(10)


if __name__ == "__main__":
    main()
