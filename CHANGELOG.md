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

## v30 — 2026-07-27 (`sha256:f13f2f3854d4…`)
- **EXL3 env knobs registered in `envs.py`** (PR #139 `00787eea`): startup unknown-var warnings
  drop 15 → 8; knobs become compile-cache factors (one-time recompile if changed).
- **SparkInfer wheel rebuilt post-PR#79** ("perf(pcie): add exact DCP top-k owner exchange",
  merged upstream): the CUDA-IPC owner-exchange module ships in the wheel. Dormant in this
  image — #79 has zero overlap with the EXL3/MoE lane and the vLLM-side owner algorithm is not
  in the pinned base yet; included so the wheel stays a strict superset of upstream master.
- Gates on the pinned digest: boot with `VLLM_EXL3_TRELLIS_MIN_M` unset PASS, warnings 15→8,
  0 capture errors, inference PASS, tool calls 4/4.

## v31 — 2026-07-27 (`sha256:0433ae9466…`)
- **Base refresh** to the unified v20 image (`sic3828fd`, digest-verified against the release
  checklist). vLLM pin unchanged → the 13-file overlay is byte-identical to v30.
- **Wheel rebuilt on the integration pin** (`#49` × 11 commits on `c3828fd`) and
  **superset-verified**: all 168 canonical SparkInfer source files present before the wheel
  overwrites the base install. This check is now part of the build procedure.
- **v30 correction:** v30's wheel was built from SparkInfer *master* rather than the base's
  integration pin, so it dropped the integration-only PCIe calibration commits from the
  installed tree. No effect on these configs (explicit DCP pins; calibration engages only on
  `auto`) — helper/auto users should move to v31.
- Gates on the pinned digest: MIN_M-unset boot PASS, warnings 8, 0 capture errors,
  inference PASS, tools 4/4.
