#!/usr/bin/env python3
"""Non-destructive release checks for the PolyTrace benchmark suite.

The script is intentionally conservative: it flags files that should not be
published in an open-source benchmark package, but never deletes anything.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path


FORBIDDEN_DIR_NAMES = {
    ".git",
    ".cache",
    ".pytest_cache",
    "__pycache__",
    "wandb",
    "outputs",
    "ray_results",
}
FORBIDDEN_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".index",
    ".parquet",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
    ".pyc",
    ".safetensors",
    ".wandb",
}
SENSITIVE_FILENAMES = {".env", ".netrc"}
TEXT_SUFFIXES = {
    ".cfg",
    ".config",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".rst",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
    ".log",
}
CONTENT_PATTERNS = {
    "private_path": re.compile(
        r"/mnt/shared-storage-user|/cpfs\d*|/cpfs01|ailab-sys|aliyun_data|zhoujiecheng"
    ),
    "wandb_api_key": re.compile(r"WANDB_API_KEY\s*=\s*[A-Za-z0-9]{20,}"),
    "generic_secret": re.compile(
        r"(api[_-]?key|secret|token)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{20,}",
        re.IGNORECASE,
    ),
}


@dataclass
class Finding:
    severity: str
    kind: str
    path: str
    detail: str
    line: int | None = None


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def add(
    finding_list: list[Finding],
    severity: str,
    kind: str,
    path: Path,
    root: Path,
    detail: str,
    line: int | None = None,
) -> None:
    finding_list.append(Finding(severity, kind, rel(path, root), detail, line))


def scan_content(path: Path, root: Path, findings: list[Finding], max_text_scan_bytes: int) -> None:
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return
    try:
        size = path.stat().st_size
    except OSError:
        return
    if size > max_text_scan_bytes:
        add(
            findings,
            "warning",
            "large_text_not_scanned",
            path,
            root,
            f"file is larger than content scan limit ({size} bytes)",
        )
        return
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line_no, line in enumerate(f, 1):
                for name, pattern in CONTENT_PATTERNS.items():
                    if pattern.search(line):
                        add(findings, "error", name, path, root, "matched sensitive content pattern", line_no)
    except OSError as exc:
        add(findings, "warning", "read_error", path, root, str(exc))


def scan(root: Path, max_size_mb: int, max_text_scan_mb: int) -> list[Finding]:
    findings: list[Finding] = []
    max_size_bytes = max_size_mb * 1024 * 1024
    max_text_scan_bytes = max_text_scan_mb * 1024 * 1024

    for current, dirnames, filenames in os.walk(root):
        current_path = Path(current)
        for dirname in list(dirnames):
            dir_path = current_path / dirname
            if dirname in FORBIDDEN_DIR_NAMES:
                add(findings, "error", "forbidden_dir", dir_path, root, f"directory '{dirname}' should not be released")
        for filename in filenames:
            path = current_path / filename
            suffix = path.suffix.lower()
            if filename in SENSITIVE_FILENAMES:
                add(findings, "error", "sensitive_filename", path, root, f"file '{filename}' may contain credentials")
            if suffix in FORBIDDEN_SUFFIXES:
                add(findings, "error", "forbidden_file_type", path, root, f"suffix '{suffix}' should not be released")
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size > max_size_bytes:
                add(findings, "error", "large_file", path, root, f"{size / 1024 / 1024:.1f} MiB exceeds {max_size_mb} MiB")
            scan_content(path, root, findings, max_text_scan_bytes)

    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate benchmark_suite before open-source release.")
    parser.add_argument("--root", default=".", help="benchmark_suite root directory")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of human-readable text")
    parser.add_argument("--max-size-mb", type=int, default=50, help="maximum file size allowed in release")
    parser.add_argument("--max-text-scan-mb", type=int, default=5, help="maximum text file size to scan for secrets")
    parser.add_argument("--max-findings", type=int, default=200, help="maximum findings to print in text mode")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    findings = scan(root, args.max_size_mb, args.max_text_scan_mb)
    errors = sum(1 for item in findings if item.severity == "error")
    warnings = sum(1 for item in findings if item.severity == "warning")

    if args.json:
        print(json.dumps({"root": str(root), "errors": errors, "warnings": warnings, "findings": [asdict(f) for f in findings]}, indent=2))
    else:
        print(f"PolyTrace release validation: {errors} error(s), {warnings} warning(s)")
        for finding in findings[: args.max_findings]:
            loc = f"{finding.path}:{finding.line}" if finding.line else finding.path
            print(f"[{finding.severity}] {finding.kind}: {loc} - {finding.detail}")
        if len(findings) > args.max_findings:
            print(f"... {len(findings) - args.max_findings} more finding(s) hidden; rerun with --max-findings {len(findings)}")

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
