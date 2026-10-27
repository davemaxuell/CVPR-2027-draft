#!/usr/bin/env bash
set -euo pipefail
scalevln_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${scalevln_root}/env.sh"
exec "${SCALEVLN_PYTHON}" "${scalevln_root}/tools/run.py" "$@"
