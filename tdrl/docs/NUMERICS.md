# Directed-head numerical checks

MRNFixed uses Euclidean symmetric distance (sym_p=1), not squared Euclidean distance, plus
max-positive directed residual. IQE computes the union length of directed intervals per
component, then a learned convex combination of mean and maximum. Shared state projections
preserve triangle inequalities by composition. No unconstrained pairwise MLP is used.

Control tests check random triples, zero self-distance, nonnegative finite values, actual
asymmetry and finite training gradients. FP32 tolerance: atol=1e-5, rtol=1e-5.
CPU BF16 tolerance: atol=2e-3, rtol=2e-2 (endpoint/linear rounding; IQE sweep arithmetic FP32).
Symmetric heads are explicitly rejected by the asymmetry check regardless of geometry, and
gross triangle violations remain failures. These tolerances are not evidence of H100 stability;
Gate 3 must retain the full distance/error distribution and observed BF16 gradient behavior
on new, independent states. A trained collapsed/nearly symmetric head must fail selection.

Tests use small synthetic embeddings for control flow and also full 4096-D response fixtures
for adapter shape/order/normalization. Neither constitutes real Qwen embeddings or GPU evidence.
