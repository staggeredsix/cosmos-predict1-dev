#!/bin/bash
set -e

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"

python3 scripts/deployment_webui.py "$@"

