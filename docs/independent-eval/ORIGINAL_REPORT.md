# GLM-5.2-EXL3-TR3-3.0bpw — independent benchmark results

Independent evaluation of `GLM-5.2-EXL3-TR3-3.0bpw` against the original
`zai-org/GLM-5.2` (753B-A40B, BF16) published scores. Run 2026-07-23/24 on a
self-hosted server; client harness scripts included for reproduction. A sibling
run of `madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid` used the identical harness,
prompts, and deterministic GPQA choice shuffles, so all three columns are
directly comparable.

## Results (pass@1, aggregated across repeats)

| Benchmark      | n (questions × repeats) | EXL3 3.0bpw | Hybrid MXFP8/NVFP4/NF3 | Original (Z.ai published) | ~95% CI (EXL3) |
|----------------|-------------------------|-------------|------------------------|---------------------------|----------------|
| AIME 2026      | 30 × 4 = 120            | 99.2        | 97.5                   | 99.2                      | ±1.6           |
| HMMT Feb 2026  | 33 × 4 = 132            | 95.5        | 97.0                   | 92.5                      | ±3.6           |
| GPQA Diamond   | 198 × 2 = 396           | 91.4        | 89.4                   | 91.2                      | ±2.8           |

All deltas versus BF16 are within sampling noise: no measurable reasoning
degradation was detected at 3.0 bits per weight. CI is a simple binomial
approximation; repeats of the same question are correlated, so true intervals
are somewhat wider.

## Environment

- Quant: EXL3 TR3, 3.0 bpw
- Hardware: 4× RTX PRO 6000, fp8 KV cache, MTP-3 (multi-token prediction /
  speculative decoding — affects throughput only, not output distribution)
- Engine: vLLM-compatible server reporting version `0.17.0rc1.dev4499+g60c82d972`,
  `max_model_len` 524288, OpenAI-compatible chat completions
- Reasoning arrives in `message.reasoning`; answers extracted from `message.content` only
- Client: async Python harness (scripts in this bundle), 32 concurrent
  requests total (16 GPQA + 8 AIME + 8 HMMT), all three benchmarks run simultaneously
- Aggregate throughput ~65 tok/s under mixed long-reasoning load; 8.63M
  completion tokens total over ~16 h

## Methodology

Matched to Z.ai's published eval settings from the zai-org/GLM-5.2 model card:

- Sampling: `temperature=1.0`, `top_p=0.95`
- Max generation: 163,840 tokens (math), 131,072 (GPQA); zero truncations occurred
- No thinking-effort override (server default thinking mode)
- Math prompts: Z.ai's system prompt (`Explanation: ... / Exact Answer: ... / Confidence: ...`)
- Math datasets: `MathArena/aime_2026`, `MathArena/hmmt_feb_2026`
- Math grading: `math-verify` symbolic equivalence (instead of Z.ai's GPT-5.5 judge);
  fallback chain: "Exact Answer:" line → last `\boxed{}` → none
- GPQA: `Idavidrein/gpqa` (gpqa_diamond), simple-evals/Artificial-Analysis MCQ
  template, answer options deterministically shuffled per (question, repeat) with
  the same seeds as the hybrid run, regex letter extraction
- pass@1 computed over all repeats pooled

## Incident note

Three requests stalled mid-run on dropped server connections (sockets stayed
ESTABLISHED client-side while the server no longer tracked the request). All
three hit the client's read timeout, auto-retried, and completed successfully —
zero lost or errored samples in the final data. Harness improvement for future
runs: TCP keepalives plus a tighter per-request timeout would surface this in
minutes instead of hours.

## Reproducible quirks (both quants, likely model-level)

- HMMT Q20: a common reasoning path converges to the wrong answer `1100`
  (gold `20460`) — both quants produced this identical wrong answer on some
  repeats. EXL3 went 2-of-4 on this question.
- GPQA idx 79 (dataset order): triggers extreme reasoning chains (121k tokens
  with format drift on the hybrid; clean 8.5k-token answer on EXL3 retry).

## Files

- `*_summary.json` — per-benchmark settings, aggregate scores, per-question rates
- `*_samples.jsonl` — per-sample records: gold, prediction, correctness, finish
  reason, completion tokens, wall seconds, final answer text (GPQA records carry
  only the model's answer tail — no question text is reproduced, per the gated
  dataset's terms)
- `mathbench.py`, `gpqa_bench.py`, `rerun_errors.py` — the harness (point
  `--base-url` at any OpenAI-compatible endpoint; `--api-key` for bearer auth)
