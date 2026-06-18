#!/bin/bash
set -euo pipefail

suite_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON:-python3}"

exec "${python_bin}" "${suite_dir}/suite.py" "$@"
