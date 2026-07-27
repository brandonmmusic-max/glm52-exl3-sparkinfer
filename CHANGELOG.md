# Changelog (runtime image lineage)

## v29 — 2026-07-27 (`sha256:2996b8ac37ff…`)
- **Boots with no env workarounds.** Draft/target role stamped at construction
  (`runner_type == "draft"`); draft layers auto-widen the Trellis window to
  `MIN_CAPTURABLE_TRELLIS_M=1`. Fixes the vllm #183 boot failure
  ("eager parity path entered during CUDA graph capture (m=3)").
- Blank env vars treated as unset (compose/K8s render unset as "").
- Validation: boot-gate with `VLLM_EXL3_TRELLIS_MIN_M` unset PASS; needle 9/9 spot;
  tool calls 4/4 + streaming PASS.

## v28 — 2026-07-26 (`sha256:fa4033287d6f…`)
- Role-aware runtime **owner token**: target/draft MoE scratch isolation no longer depends on
  which model file minted the quant config (16+ MTP model files shared the target's).
- **Batch-invariant arena cache key** (removed `x.shape[0]`); silent 4096-capacity fallback
  now raises.
- Needle 42/42 across `nvfp4_ds_mla` (24/24) and `fp8` (18/18) KV to ~479k real tokens.

## v27 — 2026-07-26 (`sha256:61ad3e2dee80…`)
- Rebase onto the finalized GG/SparkInfer v20 common base
  (`vllm 0c79e41` + `sparkinfer e603f74` + `flashinfer 801d57a`), digest-pinned.
- Adds the base's lossless PCIe/topology auto-calibration (explicit env still wins).

## earlier (v21–v26)
- v21: tr3 MTP-78 rank-sliced draft head merged into the checkpoint + loader fixes
  (draft quant-config hydration, rank-slice name normalization) — see vllm #139.
- v22–v26: KLD fixes, DCP prefill auto-policy, scope fix, arch-key work. See
  docs/RELEASE_TEST_SUITE.md for per-version validation.
