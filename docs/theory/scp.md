# Sequential convex programming

Sequential convex programming (SCP) solves a sequence of convex approximations to a nonlinear trajectory problem. The current reference trajectory defines each approximation. JAX supplies derivatives, and CVXPY assembles the convex subproblem.

`SCPMethod.solve()` in `trajopt.methods.stable.scp.scp_method` owns the iteration loop. The configured method selects discretization, convexification, trust-region, penalty-update, recovery, and convergence operations. These operations are shared in `trajopt.methods.common`; subproblem assembly lives in `trajopt.methods.stable.scp`.

Virtual buffers relax selected constraints. Penalty terms discourage these buffers, and AutoSCvx updates penalty weights and dual coefficients according to the selected preset. Hard constraints can still make a subproblem infeasible. Step handling and recovery depend on the preset; a line search is not universal.

Multiple shooting measures dynamics defects against numerical integration over each interval. Pseudospectral collocation measures consistency of the discretized collocation equations. A converged solve meets its configured discrete stopping criteria. Independent trajectory evaluation is needed to assess propagation error and between-node path-constraint violations.

See the research references in the repository README for the underlying methods.
