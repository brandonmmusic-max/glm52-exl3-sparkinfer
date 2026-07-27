#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export IMAGE="${IMAGE:-verdictai/glm52-exl3-sparkinfer:v30-gg-v20-envreg-pcietopk-vllm0c79e41-sie603f74-cu132-sm120a@sha256:f13f2f3854d40f56edc106071b6305f83c70389427cc527f42a6e99837e72d93}"
export MODEL_DIR="${MODEL_DIR:-$SCRIPT_DIR}"
export CACHE_DIR="${CACHE_DIR:-$HOME/.cache/glm52-exl3-sparkinfer}"
export PORT="${PORT:-8000}"
export BIND_ADDRESS="${BIND_ADDRESS:-127.0.0.1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3,1,2,0}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.96}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-524288}"
export DCP_KV_CACHE_INTERLEAVE_SIZE="${DCP_KV_CACHE_INTERLEAVE_SIZE:-64}"
export VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE="${VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE:-0}"
export VLLM_DCP_SHARD_DRAFT="${VLLM_DCP_SHARD_DRAFT:-1}"
export VLLM_DCP_GLOBAL_TOPK="${VLLM_DCP_GLOBAL_TOPK:-1}"
# --- Gilded Gnosis v20 FINAL DCP prefill auto-policy (TP4/DCP4) --------------
# Exactly what the v20 base launcher resolves from DCP_*=auto for TP=4/DCP=4.
# Set explicitly because this script calls `vllm serve` directly and so bypasses
# /usr/local/bin/serve-gilded-gnosis.sh (which normally maps DCP_* -> VLLM_*).
export VLLM_DCP_QUERY_SPLIT="${VLLM_DCP_QUERY_SPLIT:-1}"
export VLLM_B12X_MLA_CKV_GATHER="${VLLM_B12X_MLA_CKV_GATHER:-1}"
export VLLM_DCP_TOPK_OWNER_MERGE="${VLLM_DCP_TOPK_OWNER_MERGE:-1}"
export VLLM_DCP_INDEXER_SHARDS="${VLLM_DCP_INDEXER_SHARDS:-0}"
export VLLM_B12X_MLA_CKV_PREFETCH_DEPTH="${VLLM_B12X_MLA_CKV_PREFETCH_DEPTH:-1}"
export VLLM_B12X_MLA_CKV_PREFETCH_WORKSPACE_MIB="${VLLM_B12X_MLA_CKV_PREFETCH_WORKSPACE_MIB:-1024}"
# DCP_PREFILL_WORKSPACE=auto -> 1 for TP4/DCP4 (corrected workspace accounting).
export VLLM_DCP_PROJECT_BEFORE_MERGE="${VLLM_DCP_PROJECT_BEFORE_MERGE:-1}"
export VLLM_DCP_PROJECT_BEFORE_MERGE_MIN_PREFILL_TOKENS="${VLLM_DCP_PROJECT_BEFORE_MERGE_MIN_PREFILL_TOKENS:-1024}"
export VLLM_B12X_MLA_DCP_GATHER_IN_WORKSPACE="${VLLM_B12X_MLA_DCP_GATHER_IN_WORKSPACE:-1}"
export VLLM_DISABLE_SHARED_EXPERTS_STREAM="${VLLM_DISABLE_SHARED_EXPERTS_STREAM:-1}"
export VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE="${VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE:-64KB}"
export VLLM_PCIE_ONESHOT_FUSED_ADD_RMS_NORM_MAX_SIZE="${VLLM_PCIE_ONESHOT_FUSED_ADD_RMS_NORM_MAX_SIZE:-84KB}"
# REQUIRED for the tr3 MTP layer-78: the MTP-N draft issues small-m (m=1..N)
# GEMMs that must stay inside the Trellis cudagraph decode window. The previous
# default of 4 raises "eager parity path entered during CUDA graph capture (m=3)".
export VLLM_EXL3_TRELLIS_MIN_M="${VLLM_EXL3_TRELLIS_MIN_M:-}"   # unset = backend decides (v29): no workaround needed; draft window auto-widens
export ENABLE_MTP="${ENABLE_MTP:-1}"
export MTP_TOKENS="${MTP_TOKENS:-3}"
export MTP_DRAFT_SAMPLE_METHOD="${MTP_DRAFT_SAMPLE_METHOD:-greedy}"
export ENABLE_ASYNC_SCHEDULING="${ENABLE_ASYNC_SCHEDULING:-0}"
export GLM52_INDEX_TOPK_PATTERN="${GLM52_INDEX_TOPK_PATTERN:-FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS}"
# Empty = auto-profile the max KV cache at GPU_MEMORY_UTILIZATION. With the tr3
# MTP head this profiles to ~1,132,544 tokens (2.16x @ 524K ctx) on 4x RTX PRO
# 6000 at 0.96. Set a positive integer to pin a smaller KV cache instead.
export NUM_GPU_BLOCKS_OVERRIDE="${NUM_GPU_BLOCKS_OVERRIDE:-}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-3072}"
export COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-glm52-exl3-sparkinfer}"

COMPOSE_FILE="${COMPOSE_FILE:-$SCRIPT_DIR/docker-compose.yml}"
COMPOSE=(docker compose -f "$COMPOSE_FILE")

usage() {
  cat <<'EOF'
Usage: ./server.sh [start|stop|restart|logs|status|pull]

Environment overrides:
  IMAGE, MODEL_DIR, CACHE_DIR, PORT, BIND_ADDRESS, CUDA_VISIBLE_DEVICES,
  GPU_MEMORY_UTILIZATION, MAX_MODEL_LEN, DCP_KV_CACHE_INTERLEAVE_SIZE,
  VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE, VLLM_DCP_SHARD_DRAFT,
  VLLM_DCP_GLOBAL_TOPK,
  VLLM_DISABLE_SHARED_EXPERTS_STREAM,
  VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE,
  VLLM_PCIE_ONESHOT_FUSED_ADD_RMS_NORM_MAX_SIZE,
  ENABLE_MTP, MTP_TOKENS, MTP_DRAFT_SAMPLE_METHOD, ENABLE_ASYNC_SCHEDULING,
  GLM52_INDEX_TOPK_PATTERN,
  NUM_GPU_BLOCKS_OVERRIDE,
  MAX_NUM_BATCHED_TOKENS,
  COMPOSE_PROJECT_NAME, COMPOSE_FILE
EOF
}

require_runtime() {
  command -v docker >/dev/null 2>&1 || {
    echo "docker is required" >&2
    exit 1
  }
  docker compose version >/dev/null
  [[ -f "$COMPOSE_FILE" ]] || {
    echo "Compose file not found: $COMPOSE_FILE" >&2
    exit 1
  }
}

require_model() {
  [[ -f "$MODEL_DIR/config.json" ]] || {
    echo "Model config not found: $MODEL_DIR/config.json" >&2
    exit 1
  }
  [[ -f "$MODEL_DIR/model.safetensors.index.json" ]] || {
    echo "Model index not found: $MODEL_DIR/model.safetensors.index.json" >&2
    exit 1
  }
  mkdir -p "$CACHE_DIR"
}

action="${1:-start}"
require_runtime

case "$action" in
  start)
    require_model
    docker pull "$IMAGE"
    "${COMPOSE[@]}" up -d --force-recreate
    echo "Starting on http://localhost:$PORT"
    echo "Follow startup with: $0 logs"
    ;;
  stop)
    "${COMPOSE[@]}" down
    ;;
  restart)
    require_model
    docker pull "$IMAGE"
    "${COMPOSE[@]}" up -d --force-recreate
    echo "Restarting on http://localhost:$PORT"
    ;;
  logs)
    "${COMPOSE[@]}" logs --tail 100 -f glm52
    ;;
  status)
    "${COMPOSE[@]}" ps
    curl -fsS "http://localhost:$PORT/v1/models" || true
    printf '\n'
    ;;
  pull)
    docker pull "$IMAGE"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
