# GLM-5.2 EXL3-TR3 — production serving stack for SM120

Production-grade serving of **GLM-5.2 (753B MoE)** quantized to **EXL3 Trellis 3.0 bpw** with a
rank-sliced **EXL3 MTP-78 draft head**, on 4× NVIDIA RTX PRO 6000 Blackwell (SM120, 96 GB,
PCIe — no NVLink). TP4 / DCP4, `nvfp4_ds_mla` 4-bit KV cache, MTP-3 speculative decoding.

**Context geometry** (three different numbers — don't conflate them):
- **Model-native window: 1,048,576 tokens** (visible at boot: `Overriding draft model max model
  len from 1048576 to 524288`).
- **Shipped per-request cap: 524,288** (`--max-model-len`) — a deliberate choice so the KV pool
  holds ~2 full-length requests concurrently. Raise it toward 1M only if single-stream is
  acceptable: the pool itself is the binding limit.
- **Measured KV pool (`nvfp4_ds_mla`, util 0.96, tr3 MTP-78 head): 959,744–1,132,544 tokens**
  across boots (1.83×–2.16× concurrency at 524,288). The exact figure depends on what else is
  resident on the GPUs when vLLM profiles; read yours from the boot line
  `GPU KV cache size: N tokens`. The tr3 draft head is what buys this pool — roughly +66% KV
  versus the BF16 head.

Everything here is **reproducibly pinned**: the runtime image by registry digest, its base by
digest, and the EXL3 source layer by the two upstream PRs it is built from.

| Artifact | Pin |
|---|---|
| Runtime image | `verdictai/glm52-exl3-sparkinfer:v29-gg-v20-mincapturable-vllm0c79e41-sie603f74-cu132-sm120a` |
| Image digest | `sha256:2996b8ac37ff126a8aeebaa24df72e2154a2a1573df41f99eb48a4275e33eb41` |
| Common base (GG/SparkInfer v20) | `voipmonitor/vllm:gilded-gnosis-v20-vllm0c79e41-sie603f74-fi801d57a-cu132-20260726` @ `sha256:10261c7d65101c8aba2ce1fb59eabe73aff9d35eca5043b330cc0ce76d3c98d0` |
| Model weights | [brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw](https://huggingface.co/brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw) (includes the tr3 MTP layer-78 head) |
| EXL3 vLLM backend | [local-inference-lab/vllm #139](https://github.com/local-inference-lab/vllm/pull/139) |
| EXL3 fused-MoE kernels | [local-inference-lab/sparkinfer #49](https://github.com/local-inference-lab/sparkinfer/pull/49) |

## Quickstart

```bash
# 1. weights (~316 GB)
huggingface-cli download brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw --local-dir ./GLM-5.2-EXL3-TR3-3.0bpw

# 2. serve (compose picks up the digest-pinned image)
cd deploy
MODEL_DIR=$(realpath ../GLM-5.2-EXL3-TR3-3.0bpw) ./server.sh

# 3. verify
curl -s localhost:9200/v1/models | jq .
```

The OpenAI-compatible endpoint (chat, tools, streaming, reasoning split via the `glm45`
parser) comes up on `:9200`. First boot compiles kernels (~2 min extra); later boots hit the
compile cache.

As of **v29 no environment workarounds are required** — in particular you do **not** need
`VLLM_EXL3_TRELLIS_MIN_M=1` anymore. The backend stamps each MoE layer's draft/target role at
construction and widens the draft Trellis window itself (fixes the
[boot failure](https://github.com/local-inference-lab/vllm/issues/183) where CUDA-graph capture
of the MTP draft died with *"eager parity path entered during CUDA graph capture (m=3)"*).

## What's validated

Full methodology and raw numbers: [`docs/RELEASE_TEST_SUITE.md`](docs/RELEASE_TEST_SUITE.md).
Headlines, all measured on this exact image/digest on 4× RTX PRO 6000 (power-capped 300 W):

| Check | Result |
|---|---|
| Needle-in-a-haystack, `nvfp4_ds_mla` KV | **24/24** — depths 0.1/0.5/0.9 at 8k→480k targets, up to **479,396 real prompt tokens** |
| Needle-in-a-haystack, `fp8` KV (same everything else) | **18/18** to 479,395 tokens |
| Boot with `VLLM_EXL3_TRELLIS_MIN_M` unset | **PASS** (pre-v29: guaranteed startup failure) |
| Tool calling — non-streaming (auto / required / round-trip / negative) | **4/4** |
| Tool calling — streaming deltas (`glm47` parser) | **PASS**, no content leakage |
| Fused Trellis MoE at m=1,2,3 (NaN-poisoned arena, production tile geometry) | **bitwise-correct**, 0 FAIL / 27 |
| CUDA-graph capture + replay below plan capacity (m ∈ {1,2,3} vs cap 32) | **bit-for-bit vs eager**, stable across replays |
| KV pool at `util 0.96` (tr3 MTP-78 head) | **959,744 / 998,400 / 1,132,544 tokens** measured across boots (busy → clear → idle GPUs); 1.83×–2.16× at 524,288 |

Independent evaluation: [`docs/independent-eval/ORIGINAL_REPORT.md`](docs/independent-eval/ORIGINAL_REPORT.md).
Benchmark session logs: [`docs/benchmarks/`](docs/benchmarks/).

## Repository layout

```
deploy/                  ready-to-run serving
  docker-compose.yml       TP4/DCP4 production config (digest-pinned image)
  docker-compose-dcp1.yml  DCP1 variant: ~2× prefill throughput, ~3.4× smaller KV pool
  server.sh                convenience launcher (env-overridable)
build/                   how the image is made
  Dockerfile               thin overlay on the pinned GG/SparkInfer v20 base
  overlay/                 12 vLLM runtime files = base files + EXL3 edits (source: PR #139)
  glm52_nvfp4_mla_outer_scales.json  calibrated MLA outer scales for the nvfp4 KV cache
  extract_assets.sh        recover prebuilt binaries (exllamav3 ext, ABI shim, wheel)
                           from the published image — keeps this repo source-only
tests/                   the validation harnesses used for the results above
  needle_haystack_test.py  long-context needle probe (depths × context sweep)
  tool_call_test.py        4-scenario OpenAI tools test
  tool_call_stream_test.py streaming tool-call delta test
  validate.sh              one-shot smoke: health → inference → tools
docs/                    published results
  RELEASE_TEST_SUITE.md    full test-suite results (KLD, decode/prefill, needle, boot gates)
  independent-eval/        third-party evaluation report
  benchmarks/              dated benchmark sessions
```

## DCP4 vs DCP1 — pick your trade

Measured on this checkpoint (same image family, 300 W caps):

| | DCP4 (default) | DCP1 |
|---|---|---|
| Prefill 8k / 64k / 128k (tok/s) | ~1.45k / 1.23k / 1.17k | **2.60k / 2.42k / 2.30k** |
| KV pool (measured) | **959,744–1,132,544 tokens** | 293,760 tokens |
| Per-request cap | 524,288 shipped (model native 1,048,576) | ~262 k practical (vLLM's own DCP1 estimate: 294,976 ceiling) |

DCP shards the KV cache across ranks (capacity) at the cost of prefill collectives (speed).
Agent-style workloads dominated by prefill may prefer `deploy/docker-compose-dcp1.yml`; note the
DCP4 prefill column predates the current base's DCP prefill auto-policy work, so treat the gap as
indicative, not a controlled single-variable measurement.

## Building the image yourself

The image is a **thin overlay** — no source compile:

```bash
cd build
./extract_assets.sh          # pulls the 3 prebuilt binaries out of the published image
docker build -t glm52-exl3-sparkinfer:local .
```

`overlay/` mirrors the EXL3 source layer exactly as proposed upstream in
[vllm #139](https://github.com/local-inference-lab/vllm/pull/139); the SparkInfer wheel is built
from [sparkinfer #49](https://github.com/local-inference-lab/sparkinfer/pull/49)
(`pip wheel --no-deps .` on the PR branch). When those PRs merge, the overlay shrinks to nothing
and this repo becomes deploy-and-docs only.

**Topology note for contributors:** the two PRs target their repos' own base branches; the
*image* pins the v20 integration base by digest. Don't rebase one onto the other — the pins are
not ancestors of the PR bases (that mistake temporarily bloated #139 by ~30 foreign commits
before being caught).

## Running the validation suite

```bash
tests/validate.sh                    # health + greedy inference + tools (non-stream & stream)
python3 tests/needle_haystack_test.py --contexts 8000,65000,128000 --depths 0.1,0.5,0.9
```

The needle harness plants a unique code at fractional depths in filler text and retrieves it
greedily — misses or garbled output at long context with short-context passes is the corruption
signature it exists to catch. Run it against any config change before trusting the change.

## Known limitations

- `--max-num-seqs 8` in the shipped compose: >8 concurrent clients queue (raise it if your KV
  budget allows).
- SM120 has no TMEM/TCGEN05/WGMMA: sparse-MLA kernel families that require them (FlashMLA-sparse
  etc.) are unavailable; this stack's `B12X_MLA_SPARSE` backend + `nvfp4_ds_mla`/`fp8`/`bf16` KV
  is the working path.
- Startup logs may still warn `Unknown vLLM environment variable` for a handful of base-runtime
  knobs; these are cosmetic (the consuming code reads the environment directly). The EXL3-owned
  ones are registered as of PR #139 commit `00787eea`.
- fp8 KV works on SM120 in this stack (18/18 needle above) — ignore older notes claiming
  otherwise; nvfp4 remains the default for the ~2× larger pool at equal quality
  (KLD delta ≈ 0.015, see the test suite).

## Contributing

PRs welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). Rough split: **runtime/kernel changes**
belong upstream (vllm #139 / sparkinfer #49 lineage); **deployment, docs, validation harnesses,
and results** belong here. Every performance or correctness claim in a PR should come with the
command that produced it and, where possible, a `tests/` harness addition — that's the standard
the existing results were held to, including the ones that refuted our own assumptions.

## Acknowledgments

- [local-inference-lab](https://github.com/local-inference-lab) — the Gilded Gnosis / SparkInfer
  v20 common base and review of the EXL3 PRs
- malaiwah — the LDLQ-calibrated EXL3-TR3 rank-sliced MTP-78 head
- turboderp's [exllamav3](https://github.com/turboderp-org/exllamav3) — the EXL3 format and kernels
- Zhipu AI — GLM-5.2 (model weights under their license; this repo covers serving code only)

## License

Repository contents (scripts, configs, docs): [MIT](LICENSE). Model weights, the base image, and
upstream projects carry their own licenses.
