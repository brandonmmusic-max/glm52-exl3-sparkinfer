# GLM-5.2 EXL3 Trellis on the FINALIZED Gilded Gnosis v20 common base (2026-07-25).
# Base = canonical vLLM 0c79e41 + SparkInfer e603f74 + FlashInfer 801d57a,
#        CUTLASS DSL 4.6.0, CUDA 13.2, PyTorch 2.12, NCCL 2.30.4,
#        InstantTensor BUFFERED, DCP prefill auto-policy + workspace accounting,
#        long-context MTP alignment, deterministic dynamic-MoE output.
#        Resolves the >8k DCP prefill collapse seen in earlier v20 candidates.
#
# Per the v20 release note, EXL3 is enabled by rebasing the EXL3 *source layer*
# onto this pinned base -- NOT by pointing the base tag at an EXL3 checkpoint
# (the base packages no EXL3/Trellis loader or kernels).
FROM voipmonitor/vllm:gilded-gnosis-v20-vllm0c79e41-sie603f74-fi801d57a-cu132-20260726

LABEL org.opencontainers.image.title="GLM-5.2 EXL3 Trellis (Gilded Gnosis v20 final base)" \
      org.opencontainers.image.description="v20 finalized common runtime (vLLM 0c79e41 + SparkInfer e603f74) + EXL3 Trellis MoE overlay, rebased PRs #139/#49, MTP-78 tr3 draft head, per-model rank-sliced runtime scoping (target/draft scratch isolation), per-ordinal architecture cache key" \
      ai.verdict.base="voipmonitor/vllm:gilded-gnosis-v20-vllm0c79e41-sie603f74-fi801d57a-cu132-20260726" \
      ai.verdict.vllm.exl3="rebased-#139-on-0c79e41 + MIN_CAPTURABLE_TRELLIS_M (boots without the MIN_M=1 workaround) + role-aware owner token + batch-invariant arena key" \
      ai.verdict.sparkinfer.exl3="rebased-#49-on-e603f74 + per-ordinal arch cache key + device-validation fixes" \
      ai.verdict.model="brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw"

# 1. Rebased SparkInfer EXL3-Trellis wheel (superset of the base's canonical be0edca).
COPY dist/sparkinfer-1.0.1-py3-none-any.whl /tmp/sparkinfer-1.0.1-py3-none-any.whl
RUN /opt/venv/bin/pip install --no-deps --force-reinstall /tmp/sparkinfer-1.0.1-py3-none-any.whl \
    && rm /tmp/sparkinfer-1.0.1-py3-none-any.whl

# 2. EXL3 vLLM overlay (12 runtime files) over the canonical v20 vLLM tree.
#    Each file is the v20-final version with the EXL3 edits replayed on top, so
#    v20's DCP/prefill/indexer work is preserved (deepseek_v2.py and indexer.py
#    are the only two the new base changed; both were re-derived, not overwritten).
#    deepseek_mtp.py + models/utils.py carry the MTP-78 tr3 rank-sliced draft fixes.
COPY overlay/vllm/ /opt/venv/lib/python3.12/site-packages/vllm/
RUN /opt/venv/bin/python3 -m py_compile \
        /opt/venv/lib/python3.12/site-packages/vllm/config/model.py \
        /opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/routed_experts.py \
        /opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/__init__.py \
        /opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/exl3.py \
        /opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/deepseek_mtp.py \
        /opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/deepseek_v2.py \
        /opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/glm4_moe.py \
        /opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/utils.py \
        /opt/venv/lib/python3.12/site-packages/vllm/models/deepseek_v32/nvidia/attention.py \
        /opt/venv/lib/python3.12/site-packages/vllm/v1/attention/backends/mla/indexer.py \
        /opt/venv/lib/python3.12/site-packages/vllm/v1/core/sched/scheduler.py \
        /opt/venv/lib/python3.12/site-packages/vllm/v1/structured_output/__init__.py

# 3. EXL3 runtime assets (base has no EXL3; NCCL 2.30.4 already in base).
RUN install -d /opt/glm52/lib /opt/glm52/kv-scale
COPY exllamav3_ext.bounds-guard-sm120.cpython-312-x86_64-linux-gnu.so \
     /opt/glm52/lib/exllamav3_ext.cpython-312-x86_64-linux-gnu.so
COPY libexl3_torch212_compat.so /opt/glm52/lib/libexl3_torch212_compat.so
COPY glm52_nvfp4_mla_outer_scales.json \
     /opt/glm52/kv-scale/glm52_nvfp4_mla_outer_scales.json

ENV VLLM_EXL3_EXT_PATH=/opt/glm52/lib \
    VLLM_EXL3_ABI_SHIM=/opt/glm52/lib/libexl3_torch212_compat.so \
    VLLM_NVFP4_MLA_SCALES_FILE=/opt/glm52/kv-scale/glm52_nvfp4_mla_outer_scales.json
