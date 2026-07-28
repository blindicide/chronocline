# Mathematical model

For a finite selected-delay alphabet `D={d_i}`, independent jitter `Z`, and a uniform quantizer `Q`, the memoryless model is `Y=Q(X+Z)`. Rows of `W[i,j]=P(Y=y_j|X=d_i)` use exact CDF differences and retain lower/upper overflow mass. Capacity is `max_p I(X;Y)`.

The constrained fixed-alphabet problem maximizes mutual information subject to `D_KL(pW || P0) <= epsilon` and `p·D <= L`; `P0` must be observed through the same measurement process. In cumulative-timestamp simulations, `S_i=sum(X_k)`, `A_i=S_i+Z_i`, `R_i=Q(A_i)`, and `Y_i=R_i-R_(i-1)`. These outputs have memory and reported information quantities are empirical estimates, not exact capacity.
