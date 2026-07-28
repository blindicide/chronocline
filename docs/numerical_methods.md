# Numerical methods

Memoryless matrices use direct jitter-CDF interval probabilities, global output support, and overflow bins; no tail mass is discarded. Random phase uses deterministic Gauss–Legendre quadrature. Blahut–Arimoto terminates on an upper/lower capacity gap. Fixed-alphabet constraints use multi-start SLSQP and are accepted only after feasibility checks. Alphabet differential evolution is labelled best-found; binary grids are exact only on their stated grid.

Monte Carlo is an independent validation path, not a replacement for analytical construction. Plugin empirical information estimates should be interpreted with finite-sample caution; block estimates warn when support is large relative to trace length.

## Corrected tail handling

Upper tails use survival-function differences rather than `1-CDF`; mixtures use
log-sum-exp of component log-survival probabilities. Conditional truncation
rejects outside-support samples and records retained mass rather than looking
for nonexistent overflow labels.
