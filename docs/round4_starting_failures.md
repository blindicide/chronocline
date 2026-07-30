# Round 4 starting audit — 2026-07-30

Audited from Round-3 head `d6d26bf5f79aedc46fc46d3a290169c7b34f2fe2` on
`fix/scientific-correctness-round-3`, before Round-4 source changes.

## Commands executed

- `uv run ruff check .` — passed.
- `uv run mypy src/chronocline` — passed.
- `uv run pytest -q --cov=chronocline --cov-fail-under=88` — 56 passed,
  88.02% aggregate coverage.
- All `configs/ci/*.yaml` experiments were invoked, followed by inspection of
  existing publication directories.

## Failures and obsolete evidence

- `configs/ci/jitter_comparison.yaml` fails in
  `stable_interval_probability` with `RuntimeError: CDF and survival interval
  calculations materially disagree`. This reproduces the Laplace/Student-tail
  tolerance defect: tiny floating CDF/SF differences are treated as fatal.
- Legacy publication output is accepted by the old strict validator despite
  obsolete source provenance (including historical `80149ae...` evidence),
  stale detector KL semantics, old batching/MM rows, and degenerate phase
  configurations. This is a validator failure, not a claim that the old data
  are scientifically current.
- Current work planning does not canonically represent Cartesian sweep
  expansion or finalisation counts; the capacity-surface 2×2 regression remains
  untested at this starting point.
- Nearest-quantizer matrix construction is implemented as a shifted-floor
  approximation, so its labels/intervals can disagree with Monte Carlo.
- Stateful direct batching currently starts from timestamp quantization, and
  the Miller--Madow mutual-information correction has the wrong sign.
- Current phase/random-phase API lacks the required explicit scientific mode
  distinctions; current phase CI is only a degenerate two-point sweep.

Generated CI outputs are development evidence only and are not publication
results. Existing publication results are preserved pending the mandated legacy
migration after source corrections.
