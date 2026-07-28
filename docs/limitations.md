# Limitations

The memoryless model ignores temporal dependence. KL divergence is not a complete operational detectability measure. Empirical block mutual information is not exact channel capacity. Alphabet search can be nonconvex, synthetic jitter may not model real networks, timestamp quantization differs from operating-system batching, and finite traces bias entropy estimates. Best-found constrained alphabets are not global optima.

Random-phase semantics are explicit: fixed known phase is a DMC, per-trace
phase is latent state, and unknown-per-symbol phase is rejected. v0.6.0 result
directories are immutable historical artifacts and must not be overwritten.
