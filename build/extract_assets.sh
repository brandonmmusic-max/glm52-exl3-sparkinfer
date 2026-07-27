#!/usr/bin/env bash
# Recover the prebuilt binary assets from the published, digest-pinned image so the
# Dockerfile can rebuild without this repo carrying binaries in git.
#   - exllamav3 CUDA extension (SM120 bounds-guard build, ~121 MB)
#   - torch-2.12 ABI compatibility shim
#   - rebased SparkInfer wheel (also buildable from sparkinfer PR #49: pip wheel --no-deps .)
set -euo pipefail
IMG="${IMG:-verdictai/glm52-exl3-sparkinfer:v29-gg-v20-mincapturable-vllm0c79e41-sie603f74-cu132-sm120a@sha256:2996b8ac37ff126a8aeebaa24df72e2154a2a1573df41f99eb48a4275e33eb41}"
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
