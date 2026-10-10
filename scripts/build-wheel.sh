#!/usr/bin/env bash
# Build OpenViking wheel in container (NixOS host can't compile natively)
# Usage: ./scripts/build-wheel.sh [--install]
# Optional mirrors:
#   APT_MIRROR=mirrors.ustc.edu.cn ./scripts/build-wheel.sh
#   UV_INDEX_URL=https://mirrors.ustc.edu.cn/pypi/web/simple ./scripts/build-wheel.sh
set -euo pipefail

cd "$(dirname "$0")/.."
DIST_DIR="./dist"

# Accept either a full URL or a mirror hostname.
APT_MIRROR="${APT_MIRROR:-https://deb.debian.org}"
if [[ "$APT_MIRROR" != http://* && "$APT_MIRROR" != https://* ]]; then
  APT_MIRROR="https://$APT_MIRROR"
fi
APT_MIRROR="${APT_MIRROR%/}"
UV_INDEX_URL="${UV_INDEX_URL:-https://pypi.org/simple}"

# Clean old wheels
rm -f "$DIST_DIR"/openviking-*.whl "$DIST_DIR"/openviking-*.tar.gz

echo "Building wheel in container..."
echo "APT mirror: $APT_MIRROR"
echo "Python index: $UV_INDEX_URL"
podman run --rm \
  -v .:/src \
  -w /src \
  --net host \
  -e "APT_MIRROR=$APT_MIRROR" \
  -e "UV_INDEX_URL=$UV_INDEX_URL" \
  rust:latest bash -c '
    set -euo pipefail
    rm -f /etc/apt/sources.list.d/debian.sources
    printf "%s\\n" \
      "deb ${APT_MIRROR}/debian trixie main" \
      "deb ${APT_MIRROR}/debian trixie-updates main" \
      "deb ${APT_MIRROR}/debian-security trixie-security main" \
      > /etc/apt/sources.list
    apt-get update
    apt-get install -y python3-dev python3-pip > /dev/null
    pip install --break-system-packages --index-url "$UV_INDEX_URL" uv
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_OPENVIKING=0.0.1.dev0 \
      uv build --python-preference system
  '

shopt -s nullglob
wheels=("$DIST_DIR"/openviking-*.whl)
shopt -u nullglob
if (( ${#wheels[@]} == 0 )); then
  echo "ERROR: no wheel produced" >&2
  exit 1
fi
WHEEL="${wheels[0]}"

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
