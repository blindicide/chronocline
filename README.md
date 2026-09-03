> **Archive Notice:** This repository is archived. Completed research project consolidated into [blindicide/sci-archive](https://github.com/blindicide/sci-archive) under `chronocline/`.

# Project Chronocline

Project Chronocline is a research-grade Python framework for modelling quantized network timing channels. It studies deliberately selected inter-packet delays observed through random jitter, finite timestamp resolution, batching, and statistical detection constraints. It is a mathematical simulator, not a network traffic tool.

The memoryless core is `Y = Q(X + Z)`, with a finite delay alphabet, a jitter distribution, and an explicit timestamp quantizer. The package computes channel matrices, mutual information, capacities, constrained optima, detector metrics, and stateful timestamp simulations.

## Installation and quick start

```bash
python -m pip install -e '.[dev]'
chronocline validate-config configs/smoke.yaml
chronocline experiment configs/smoke.yaml
chronocline validate-results results/smoke/latest
```

All results contain the resolved configuration, manifest, environment information, diagnostics, tables, and figures. Repeated runs with a fixed seed use deterministic child streams. See `docs/experiment_protocol.md` for reproduction commands.

## Scope and safety

The project generates and analyses synthetic or user-supplied timing data only. It does not inject packets, capture interfaces, target hosts, or provide operational traffic-evasion tooling.

## Development

```bash
pytest
ruff check .
mypy src/chronocline
```

Citation metadata is in `CITATION.cff`.
