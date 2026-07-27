# Contributing

PRs are welcome. A few rules keep this repo trustworthy:

## Where changes belong
- **Runtime / kernel / loader code** → upstream, on the lineage of
  [vllm #139](https://github.com/local-inference-lab/vllm/pull/139) and
  [sparkinfer #49](https://github.com/local-inference-lab/sparkinfer/pull/49).
  `build/overlay/` mirrors that work; don't fork it here.
- **Deployment configs, docs, validation harnesses, results** → this repo.

## Standards
1. **Pin by digest.** Any image reference in a PR must be `tag@sha256:...`.
2. **Claims come with commands.** A performance or correctness claim needs the exact
   reproduction command and environment (GPU model, power limit, DCP/TP, KV dtype).
3. **Negative results are results.** If you refute something in the docs, that's a PR too.
4. **Validation before assertion.** Run `tests/validate.sh` against your change; for anything
   touching attention/KV/decoding, run the needle harness at ≥3 context lengths and report
   the table. "Short prompts look fine" is not evidence — the failure modes this stack has
   actually seen were invisible below ~1k tokens.
5. **One variable at a time** where possible; when confounded, say so in the PR body.

## Topology (for anyone touching build/)
The upstream PRs target their repos' base branches; the image pins the v20 integration base by
digest. These are different lineages on purpose — rebase the PRs onto their own bases, derive
the overlay from the pin. Mixing them has bitten before.
