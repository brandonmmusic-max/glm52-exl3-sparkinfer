# 2026-07-27 — v29 decode matrix (c1–c32, 0–32k ctx) + fresh DCP4 prefill

Image: `verdictai/glm52-exl3-sparkinfer:v29-…@sha256:2996b8ac37ff…` (the pinned production
image). 4× RTX PRO 6000 Blackwell, 300 W caps, TP4/DCP4, `nvfp4_ds_mla` KV, MTP-3,
`max-num-seqs 8` (shipped default), temperature 0, duration 20 s/cell, `max_tokens` 8192.

## Sustained decode — aggregate tok/s

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| 0   | 82.9 | 140.0 | 212.0 | **302.5** | capped (8/16) | capped (8/32) |
| 8k  | 78.7 | 134.8 | 198.5 | 288.2 | capped | capped |
| 16k | 80.3 | 130.0 | 199.2 | 272.4 | capped | capped |
| 32k | 77.4 | 136.9 | 205.3 | **292.5** | capped | capped |

- c16/c32 cells are reported as **capacity-capped**, not as throughput: the shipped compose
  pins `--max-num-seqs 8`, so only 8 clients are ever active and the harness flags the cell
  (`∅ (8/16)`, `∅ (8/32)`) instead of printing a queueing-inflated number.
- Decode is nearly context-flat: c8 at 32k is within ~3% of c8 at zero context.

## Standalone prefill (cold profile, DCP4, this same image)

| ctx | tok/s | TTFT |
|---|---|---|
| 8K   | 2,341 | 3.50 s |
| 64K  | 1,755 | 36.8 s |
| 128K | 1,657 | 77.8 s |

Raw artifacts (in this repo, `docs/benchmarks/2026-07-27/`):
[decode log](2026-07-27/decode-c1-c32-ctx0-32k.log) ·
[decode json](2026-07-27/decode-c1-c32-ctx0-32k.json) ·
[prefill log](2026-07-27/prefill-v29-dcp4.log) ·
[prefill json](2026-07-27/prefill-v29-dcp4.json).
Harness = `llm_decode_bench.py` (sha in the release suite).
