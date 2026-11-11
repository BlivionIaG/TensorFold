#!/usr/bin/env bash
# The mlx environment of the references and Mac-alone baselines (Python 3.12, mlx 0.32.4, mlx-lm 0.32.2; tools/vega-macos/requirements-mlx.txt).
#   tools/vega-macos/setup-mlx-env.sh [DIR]      default ~/.cache/tensorfold-vega/mlxenv; needs `uv`
# The scripts that say /tmp/mlxenv/bin/python (moe_ref.py, mlx_ref.py docstrings, the study's routing_trace.py) mean this interpreter.
set -euo pipefail
V=${1:-$HOME/.cache/tensorfold-vega/mlxenv}
command -v uv >/dev/null || { echo "uv not found: brew install uv" >&2; exit 3; }
[ -x "$V/bin/python" ] || uv venv --python 3.12 "$V"
uv pip install --quiet --python "$V/bin/python" -r "$(dirname "$0")/requirements-mlx.txt"
"$V/bin/python" -c "from importlib.metadata import version as v; print('mlx env ok: $V mlx', v('mlx'), 'mlx-lm', v('mlx-lm'))"
