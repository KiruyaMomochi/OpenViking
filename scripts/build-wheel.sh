#!/usr/bin/env bash
# Build OpenViking wheel in container (NixOS host can't compile natively)
# Usage: ./scripts/build-wheel.sh [--install]
set -euo pipefail

cd "$(dirname "$0")/.."
DIST_DIR="./dist"

# Clean old wheels
rm -f "$DIST_DIR"/openviking-*.whl "$DIST_DIR"/openviking-*.tar.gz

echo "Building wheel in container..."
podman run --rm -v .:/src -w /src --net host rust:latest bash -c \
  "apt-get update && apt-get install -y python3-dev python3-pip > /dev/null && \
   pip install --break-system-packages uv && \
   SETUPTOOLS_SCM_PRETEND_VERSION_FOR_OPENVIKING=0.0.1.dev0 uv build --python-preference system"

WHEEL=$(ls "$DIST_DIR"/openviking-*.whl 2>/dev/null | head -1)
if [[ -z "$WHEEL" ]]; then
  echo "ERROR: no wheel produced" >&2
  exit 1
fi

echo "---"
echo "Built: $WHEEL"

if [[ "${1:-}" == "--install" ]]; then
  echo "Installing..."
  uv tool install --reinstall "$WHEEL"
  echo "Done. Restart OV service: systemctl --user restart openviking"
else
  echo "Install with: uv tool install --reinstall $WHEEL"
  echo "Or rerun with: ./scripts/build-wheel.sh --install"
fi
