# Experiment protocol

```bash
.venv/bin/chronocline validate-config configs/smoke.yaml
.venv/bin/chronocline experiment configs/smoke.yaml
.venv/bin/chronocline validate-results results/smoke/<run-id>
.venv/bin/chronocline plot results/smoke/<run-id> --locale ru
```

Use `chronocline experiment CONFIG --dry-run` to inspect sweep cardinality without calculations. Identical configurations map to deterministic run identifiers and completed manifests support resumption.
