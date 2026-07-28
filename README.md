# Project Chronocline

Project Chronocline is a research-grade Python framework for modelling quantized network timing channels. It studies deliberately selected inter-packet delays observed through random jitter, finite timestamp resolution, batching, and statistical detection constraints. It is a mathematical simulator, not a network traffic tool.

The memoryless core is `Y = Q(X + Z)`, with a finite delay alphabet, a jitter distribution, and an explicit timestamp quantizer. The package computes channel matrices, mutual information, capacities, constrained optima, detector metrics, and stateful timestamp simulations.

## Installation and quick start

```bash
python -m pip install -e '.[dev]'
chronocline validate-config configs/smoke.yaml
chronocline experiment configs/smoke.yaml
chronocline validate-results results/smoke
chronocline plot results/smoke
```

An experiment directory contains a portable `LATEST` file naming its most recent run; commands accept
either that directory or a concrete run directory. A completed result contains the resolved
configuration, manifest, environment information, diagnostics, scalar summaries, matrices or
experiment tables, and figures with source tables. Publication configurations refuse dirty source
trees unless explicitly overridden for development.

Schema 2.0 result bundles are not backward-compatible with the incomplete generic-runner
campaign recorded for v0.5.0. The v0.6.0 campaigns dispatch to distinct memoryless, phase,
detectability, finite-sample detection, batching, jitter-comparison, and alphabet-search runners.
Exact memoryless capacities are never used to describe stateful batching estimates.

See `docs/experiment_protocol.md` for reproduction commands, expected campaign scope, baseline
construction, and the distinction between exact and empirical metrics.

## Scope and safety

The project generates and analyses synthetic or user-supplied timing data only. It does not inject packets, capture interfaces, target hosts, or provide operational traffic-evasion tooling.

## Development

```bash
pytest
ruff check .
mypy src/chronocline
```

Citation metadata is in `CITATION.cff`.
