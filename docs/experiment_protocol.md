# Experiment protocol

```bash
.venv/bin/chronocline validate-config configs/ci/capacity_surface.yaml
.venv/bin/chronocline experiment configs/ci/capacity_surface.yaml
.venv/bin/chronocline validate-results results/ci_capacity_surface
.venv/bin/chronocline plot results/ci_capacity_surface --locale ru
```

Use `chronocline experiment CONFIG --dry-run` to inspect sweep cardinality without calculations.
The CI configurations are small executions for every experiment kind. Publication configurations
live in `configs/publication/` and require a clean source commit. Run them only after tests, Ruff,
and MyPy pass; regenerate figures from stored result CSV/table data, then validate every result
directory again.

`LATEST` is a text pointer, not a symbolic link, so result directories remain portable. A manifest
records the clean commit, configuration hash, schema and runner versions, worker count, and every
generated-file checksum. The detector compares a configured active output distribution against a
baseline passed through the same channel and quantizer; KL values are in bits. Memoryless capacity
is exact for its discrete channel model, whereas batching statistics and block mutual information
are empirical stateful estimates.

The v0.5.0 generic-runner outputs are historical and are not schema-2 publication evidence. Do not
mix them with current results. The publication campaign includes a 12-by-12 capacity surface,
33 distinct phase positions on `[0, Δ)`, 20 detectability budgets, nine finite-sample detector
sizes, matched-variance jitter families, binary and ternary alphabet search, and stateful batching.
