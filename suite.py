#!/usr/bin/env python3
"""Unified workload benchmark suite for verl-style RL repositories.

The suite intentionally depends only on the Python standard library so it can
run on login/dev machines without the training environments installed. Config
files use a YAML extension, but are written as JSON, which is valid YAML too.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
KV_RE = re.compile(r"([A-Za-z0-9_./-]+):([-+0-9.eE]+)")


@dataclass
class SuiteConfig:
    name: str
    repo: str
    repo_path: Path
    benchmark_script: Path
    workload_path: Path | None
    benchmark_log_path: Path
    formal_log_path: Path | None
    step_groups: str
    repeat: int
    env: dict[str, str]
    workload: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> "SuiteConfig":
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        repo_path = expand_path(raw["repo_path"])
        workload_path = raw.get("workload_path")
        formal_log_path = raw.get("formal_log_path")
        return cls(
            name=raw["name"],
            repo=raw["repo"],
            repo_path=repo_path,
            benchmark_script=expand_path(raw["benchmark_script"]),
            workload_path=expand_path(workload_path) if workload_path else None,
            benchmark_log_path=expand_path(raw["benchmark_log_path"]),
            formal_log_path=expand_path(formal_log_path) if formal_log_path else None,
            step_groups=str(raw.get("step_groups", "0")),
            repeat=int(raw.get("repeat", 1)),
            env={str(k): str(v) for k, v in raw.get("env", {}).items()},
            workload=dict(raw.get("workload", {})),
        )

    def benchmark_env(self) -> dict[str, str]:
        env = dict(self.env)
        env.setdefault("WORKLOAD_BENCHMARK", "1")
        env.setdefault("WORKLOAD_BENCHMARK_REPEAT", str(self.repeat))
        env.setdefault("WORKLOAD_STEP_GROUPS", self.step_groups)
        if self.workload_path is not None:
            env.setdefault("WORKLOAD_PATH", str(self.workload_path))
        return env


def expand_path(value: str) -> Path:
    return Path(os.path.expandvars(str(value))).expanduser()


def strip_ansi(line: str) -> str:
    return ANSI_RE.sub("", line)


def parse_step_groups(step_groups: str) -> list[list[str]]:
    groups: list[list[str]] = []
    for group in step_groups.split(";"):
        items = [item.strip() for item in group.split(",") if item.strip()]
        if items:
            groups.append(items)
    return groups


def parse_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
            else:
                rows.append({"value": row})
    return rows


def validate_multiturn_workload(cfg: SuiteConfig) -> dict[str, Any]:
    base = cfg.workload_path
    if base is None:
        return {"ok": False, "error": "workload_path is not configured"}

    expected = int(cfg.workload.get("expected_records_per_step", 0) or 0)
    stats: dict[str, Any] = {"type": "multiturn", "steps": {}, "ok": True}
    for group in parse_step_groups(cfg.step_groups):
        for step in group:
            path = base / f"multiturn_workload_step_{step}.jsonl"
            if not path.exists():
                stats["ok"] = False
                stats["steps"][step] = {"ok": False, "error": f"missing {path}"}
                continue
            rows = parse_jsonl(path)
            prompt_groups = {row.get("prompt_group_id") for row in rows if "prompt_group_id" in row}
            sample_ids = [row.get("sample_id") for row in rows if "sample_id" in row]
            tool_calls = sum(len(row.get("tool_calls", []) or []) for row in rows)
            turns = sum(len(row.get("turns", []) or []) for row in rows)
            ok = len(rows) > 0 and (expected <= 0 or len(rows) == expected)
            stats["ok"] = stats["ok"] and ok
            stats["steps"][step] = {
                "ok": ok,
                "records": len(rows),
                "expected_records": expected or None,
                "prompt_groups": len(prompt_groups),
                "sample_id_min": min(sample_ids) if sample_ids else None,
                "sample_id_max": max(sample_ids) if sample_ids else None,
                "turns": turns,
                "tool_calls": tool_calls,
            }
    return stats


def validate_packed_workload(cfg: SuiteConfig) -> dict[str, Any]:
    base = cfg.workload_path
    if base is None:
        return {"ok": False, "error": "workload_path is not configured"}

    stats: dict[str, Any] = {"type": "packed_lengths", "steps": {}, "ok": True}
    for group in parse_step_groups(cfg.step_groups):
        for step in group:
            path = base / f"packed_lengths_step_{step}.jsonl"
            if not path.exists():
                stats["ok"] = False
                stats["steps"][step] = {"ok": False, "error": f"missing {path}"}
                continue
            rows = parse_jsonl(path)
            ok = len(rows) > 0
            stats["ok"] = stats["ok"] and ok
            stats["steps"][step] = {"ok": ok, "records": len(rows)}

    filter_file = cfg.workload.get("filter_indices_file")
    if filter_file:
        fpath = base / str(filter_file)
        stats["filter_indices"] = {"exists": fpath.exists(), "path": str(fpath)}
        stats["ok"] = stats["ok"] and fpath.exists()
    reward_glob = cfg.workload.get("reward_timing_glob")
    if reward_glob:
        files = sorted(base.glob(str(reward_glob)))
        stats["reward_timings"] = {"files": len(files), "glob": str(reward_glob)}
    return stats


def validate_workload(cfg: SuiteConfig) -> dict[str, Any]:
    wtype = cfg.workload.get("type", "packed_lengths")
    if wtype == "multiturn":
        return validate_multiturn_workload(cfg)
    return validate_packed_workload(cfg)


def parse_metrics_dict(line: str) -> dict[str, float] | None:
    marker = "/metrics:"
    if marker not in line:
        return None
    payload = line.split(marker, 1)[1].strip()
    try:
        obj = ast.literal_eval(payload)
    except (SyntaxError, ValueError):
        return None
    if not isinstance(obj, dict):
        return None
    return {str(k): float(v) for k, v in obj.items() if isinstance(v, (int, float))}


def parse_console_kv_line(line: str) -> dict[str, float] | None:
    if "timing_s/" not in line or "step:" not in line:
        return None
    metrics: dict[str, float] = {}
    for key, value in KV_RE.findall(line):
        try:
            metrics[key] = float(value)
        except ValueError:
            pass
    return metrics or None


def parse_log(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []
    metrics: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = strip_ansi(raw_line)
            parsed = parse_metrics_dict(line) or parse_console_kv_line(line)
            if parsed:
                metrics.append(parsed)
    return metrics


def pick_timing(metrics: dict[str, float], key: str) -> float | None:
    return metrics.get(f"timing_s/{key}") or metrics.get(key)


def compare_metrics(bench: list[dict[str, float]], formal: list[dict[str, float]]) -> list[dict[str, Any]]:
    stages = ["gen", "reward", "old_log_prob", "ref", "adv", "update_actor", "step"]
    rows: list[dict[str, Any]] = []
    for idx, b in enumerate(bench):
        f = formal[idx] if idx < len(formal) else {}
        row: dict[str, Any] = {"index": idx}
        for stage in stages:
            bv = pick_timing(b, stage)
            fv = pick_timing(f, stage)
            row[f"{stage}_benchmark"] = bv
            row[f"{stage}_formal"] = fv
            row[f"{stage}_error_pct"] = ((bv - fv) / fv * 100.0) if bv is not None and fv else None
        row["benchmark_total_tokens"] = b.get("perf/total_num_tokens")
        row["formal_total_tokens"] = f.get("perf/total_num_tokens")
        rows.append(row)
    return rows


def command_for_config(cfg: SuiteConfig) -> str:
    exports = " ".join(f"{key}={shell_quote(value)}" for key, value in sorted(cfg.benchmark_env().items()))
    return f"cd {shell_quote(str(cfg.repo_path))} && {exports} bash {shell_quote(str(cfg.benchmark_script))}"


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def write_report(cfg: SuiteConfig, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    validation = validate_workload(cfg)
    benchmark_metrics = parse_log(cfg.benchmark_log_path)
    formal_metrics = parse_log(cfg.formal_log_path) if cfg.formal_log_path else []
    comparisons = compare_metrics(benchmark_metrics, formal_metrics)
    report = {
        "name": cfg.name,
        "repo": cfg.repo,
        "command": command_for_config(cfg),
        "validation": validation,
        "benchmark_metrics_count": len(benchmark_metrics),
        "formal_metrics_count": len(formal_metrics),
        "comparisons": comparisons,
    }
    json_path = out_dir / f"{cfg.name}.summary.json"
    md_path = out_dir / f"{cfg.name}.summary.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return report


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Benchmark Summary: {report['name']}",
        "",
        f"- Repo: `{report['repo']}`",
        f"- Benchmark metrics: `{report['benchmark_metrics_count']}`",
        f"- Formal metrics: `{report['formal_metrics_count']}`",
        f"- Workload valid: `{report['validation'].get('ok')}`",
        "",
        "## Command",
        "",
        "```bash",
        report["command"],
        "```",
        "",
        "## Timing Comparison",
        "",
        "| idx | gen % | old_log_prob % | ref % | update_actor % | step % |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["comparisons"]:
        lines.append(
            "| {idx} | {gen} | {old} | {ref} | {upd} | {step} |".format(
                idx=row["index"],
                gen=fmt_pct(row.get("gen_error_pct")),
                old=fmt_pct(row.get("old_log_prob_error_pct")),
                ref=fmt_pct(row.get("ref_error_pct")),
                upd=fmt_pct(row.get("update_actor_error_pct")),
                step=fmt_pct(row.get("step_error_pct")),
            )
        )
    lines.extend(["", "## Workload Validation", "", "```json", json.dumps(report["validation"], indent=2), "```", ""])
    return "\n".join(lines)


def fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.2f}%"


def discover_configs(config_dir: Path) -> list[Path]:
    return sorted(config_dir.glob("*.yaml")) + sorted(config_dir.glob("*.json"))


def run_execute(cfg: SuiteConfig) -> int:
    env = os.environ.copy()
    env.update(cfg.benchmark_env())
    return subprocess.call(["bash", str(cfg.benchmark_script)], cwd=str(cfg.repo_path), env=env)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Unified workload benchmark suite")
    parser.add_argument("configs", nargs="*", type=Path, help="Config files. Defaults to configs/*.yaml")
    parser.add_argument("--config-dir", type=Path, default=Path(__file__).with_name("configs"))
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).with_name("reports"))
    parser.add_argument("--dry-run", action="store_true", help="Print benchmark commands only")
    parser.add_argument("--validate", action="store_true", help="Validate workload files only")
    parser.add_argument("--analyze", action="store_true", help="Parse logs and write reports")
    parser.add_argument("--execute", action="store_true", help="Actually launch benchmark scripts")
    args = parser.parse_args(argv)

    config_paths = args.configs or discover_configs(args.config_dir)
    if not config_paths:
        print(f"No configs found under {args.config_dir}", file=sys.stderr)
        return 1

    configs = [SuiteConfig.load(path) for path in config_paths]
    if not any([args.dry_run, args.validate, args.analyze, args.execute]):
        args.dry_run = True
        args.validate = True
        args.analyze = True

    exit_code = 0
    for cfg in configs:
        print(f"== {cfg.name} ({cfg.repo}) ==")
        if args.dry_run:
            print(command_for_config(cfg))
        if args.validate:
            validation = validate_workload(cfg)
            print(json.dumps(validation, indent=2, ensure_ascii=False))
            if not validation.get("ok"):
                exit_code = 2
        if args.analyze:
            report = write_report(cfg, args.out_dir)
            print(f"wrote reports for {cfg.name}: {args.out_dir}")
            if report["formal_metrics_count"] == 0:
                print(f"warning: no formal metrics parsed for {cfg.name}", file=sys.stderr)
        if args.execute:
            code = run_execute(cfg)
            exit_code = exit_code or code
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
