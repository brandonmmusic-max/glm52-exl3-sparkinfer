#!/usr/bin/env bash
# Recover the prebuilt binary assets from the published, digest-pinned image so the
# Dockerfile can rebuild without this repo carrying binaries in git.
#   - exllamav3 CUDA extension (SM120 bounds-guard build, ~121 MB)
#   - torch-2.12 ABI compatibility shim
#   - rebased SparkInfer wheel (also buildable from sparkinfer PR #49: pip wheel --no-deps .)
set -euo pipefail
IMG="${IMG:-verdictai/glm52-exl3-sparkinfer:v31-gg-v20-sic3828fd-vllm0c79e41-cu132-sm120a@sha256:0433ae94665b769b78dd301f952d907508a3ba80bce47a1630ec20ade8812dff}"
cd "$(dirname "$0")"
mkdir -p dist
cid=$(docker create "$IMG")
trap 'docker rm -f "$cid" >/dev/null' EXIT
docker cp "$cid":/opt/glm52/lib/exllamav3_ext.cpython-312-x86_64-linux-gnu.so \
          exllamav3_ext.bounds-guard-sm120.cpython-312-x86_64-linux-gnu.so
docker cp "$cid":/opt/glm52/lib/libexl3_torch212_compat.so libexl3_torch212_compat.so
pyver=$(docker run --rm --entrypoint /opt/venv/bin/python "$IMG" - <<'PY'
import importlib.metadata as m; print(m.version("sparkinfer"))
PY
)
docker cp "$cid":/opt/venv/lib/python3.12/site-packages/sparkinfer /tmp/_si_pkg 2>/dev/null || true
echo "sparkinfer version in image: $pyver"
echo "NOTE: the wheel itself is rebuilt from sparkinfer PR #49 (pip wheel --no-deps .);"
echo "      place it at dist/sparkinfer-<ver>-py3-none-any.whl"
echo "assets extracted:"
ls -la ./*.so
