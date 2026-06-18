# PolyTrace Benchmark Suite

PolyTrace is a workload-driven benchmark suite for RL training systems. It
replays the shape and timing pressure of real training steps from compact
workload traces, so the expensive parts of rollout, log-prob, actor update,
reward/tool latency, and multi-turn interaction can be benchmarked without
publishing private training datasets.

This directory is intended to be open sourced together with patched benchmark
implementations for four codebases:

- `verl-video`: video RL / vision-language rollout benchmark.
- `verl`: DAPO/GRPO text RL benchmark with filtering-aware replay.
- `rllm`: coding RL benchmark with reward-time replay.
- `search_r1`: multi-turn search/tool RL benchmark.

Cluster-specific launch scripts, training datasets, logs, checkpoints, wandb
runs, retriever indexes, and raw profiling traces are intentionally ignored.
Only source code, benchmark configs, and compact benchmark inputs should be
committed.

## Layout

```text
benchmark_suite/
  configs/                 # JSON-as-YAML benchmark configs
  dataset/                 # public compact workload inputs
  tools/validate_release.py # non-destructive release checks
  verl-video/              # patched repo copy, without private scripts/data
  verl/                    # patched repo copy, without private scripts/data
  rllm/                    # patched repo copy, without private scripts/data
  search_r1/               # patched repo copy, without private scripts/data
  run_suite.sh             # suite wrapper
  suite.py                 # dry-run / validate / analyze / execute CLI
```

The config files have `.yaml` names but use JSON syntax so the suite has no
external dependency.


## Quick Start

Environments:
For verl and verl-video, use verl docker images: verl-app-verl0.4-sglang0.4.6.post5-vllm0.8.5-mcore0.12.2-te2.2
For rllm, use verl docker images: verl-ngc-th2.4.0-cu124-vllm0.6.3-te1.7-v0.0.4, and install some packages like sentence transformers. Moreover, firejail is needed to conduct sandbox validation
For search-r1, use verl docker images: verl-app-verl0.4-sglang0.4.6.post5-vllm0.8.5-mcore0.12.2-te2.2, and pip install sglang0.4.6.post4 --no-deps.

If the four patched repos live inside `benchmark_suite/`, set:

```bash
cd /path/to/PolyTrace
export POLYTRACE_WORKSPACE=$PWD/benchmark_suite
```

If you keep the repos somewhere else, set `POLYTRACE_WORKSPACE` to that parent
directory and update `configs/*.yaml` accordingly.

Run metadata-only operations on any CPU machine:

```bash
# Print commands without launching Ray/GPU jobs.
bash benchmark_suite/run_suite.sh --dry-run

# Validate workload files referenced by configs.
bash benchmark_suite/run_suite.sh --validate

# Parse existing formal/benchmark logs and write reports.
bash benchmark_suite/run_suite.sh --analyze

# Check whether the tree is safe to publish.
python3 benchmark_suite/tools/validate_release.py --root benchmark_suite
```

Reports are generated under `benchmark_suite/reports/` and are ignored by git.
They summarize existing logs; they do not mean a fresh GPU benchmark was run.

## Benchmark Launch Method

Use `--execute` only on a GPU machine with the corresponding training runtime,
model checkpoints, Ray setup, rollout backend, and optional retriever/reward
services installed:

```bash
export POLYTRACE_WORKSPACE=$PWD/benchmark_suite
bash benchmark_suite/run_suite.sh --execute benchmark_suite/configs/verl_video_qwen2_5_vl_7b.yaml
bash benchmark_suite/run_suite.sh --execute benchmark_suite/configs/verl_dapo_qwen2_5_32b.yaml
bash benchmark_suite/run_suite.sh --execute benchmark_suite/configs/rllm_deepcoder_14b_16k.yaml
bash benchmark_suite/run_suite.sh --execute benchmark_suite/configs/search_r1_qwen2_5_7b.yaml
```

The suite launches the configured `benchmark_script` with workload environment
variables. Because repo-local `scripts/` directories are ignored, each user
should keep their real cluster launch script outside git or recreate it locally,
then set `benchmark_script` in the config to that local path.

The common variables are:

- `WORKLOAD_BENCHMARK=1`
- `WORKLOAD_PATH=<compact workload directory>`
- `WORKLOAD_STEP_GROUPS=<step replay grouping>`
- `WORKLOAD_BENCHMARK_REPEAT=<repeat count>`
- `WORKLOAD_MAX_TOKEN_POLICY=<length estimation policy>`

Repo-specific variables are kept in the config `env` block.

## Manual Launch Template

The suite command is equivalent to:

```bash
cd "$POLYTRACE_WORKSPACE/<repo>"
WORKLOAD_BENCHMARK=1 WORKLOAD_PATH="$POLYTRACE_WORKSPACE/<repo>/profile/packed_length_log/<run_id>" WORKLOAD_STEP_GROUPS="0" WORKLOAD_BENCHMARK_REPEAT=1 WORKLOAD_MAX_TOKEN_POLICY=adaptive_blend bash /path/to/local_benchmark_launcher.sh
```

For `search_r1`, the workload path points to `profile/multiturn_workload_log`,
and benchmark mode may also start or contact a retriever service depending on
the local config.

## Workload Types

### Single-Turn Length Replay

Used by `verl-video` and text-only `verl`. The workload records prompt and
response length distributions. Benchmark mode creates synthetic tensors with the
same shape pressure and replays rollout, log-prob, reference, advantage, and
actor-update stages.

### Filtering-Aware DAPO Replay

DAPO can filter samples before later stages. A faithful benchmark records which
samples survived, not only how many, because each sample has a different length.
The `verl` config can validate a `dapo_filter_indices.jsonl` file.

### Coding Reward Replay

Coding RL spends time in local sandbox tests. `rllm` records reward timing and
can sleep during benchmark replay, which is usually closer to end-to-end timing
than adding reward time after the fact.

### Multi-Turn Tool Replay

`search_r1` workloads contain turn-level prompt/response lengths, tool calls,
and tool latency information. Prefix-cache behavior can dominate this workload,
so benchmark replay should preserve prompt grouping and turn order as much as
possible. Live retriever/tool replay can be enabled for end-to-end validation.

## Dataset

There are some industrial workload in model deployment, you can replay these workload or synthetic workload to benchmark various framework or cluster.