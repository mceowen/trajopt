import numpy as np
import scipy as sp

#####################################################
##### PSEUDOSPECTRAL AND POLYNOMIAL PROTOTYPES: #####
#####################################################
USE_SPARTAN = True
USE_HARDCODED_NEWTON = False

# ---------------------------------------------------------------------
# Legendre polynomial helper
# ---------------------------------------------------------------------
def compute_legendre(N: int, x: np.ndarray, use_spartan: bool = USE_SPARTAN) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the Legendre polynomial P_n(x) and its derivative P_n'(x).

    Inputs:
        N :             Polynomial order.
        x : Evaluation points.
        use_spartan :   Use SPARTAN-style Legendre recursive computation if True, scipy computation if False.

    Outputs:
        Pn :            Legendre polynomial evaluated at x.
        dPn :           Derivative d/dx P_n(x) evaluated at x.

    """
    x = np.asarray(x, dtype=float)

    if use_spartan:
        x = np.asarray(x, dtype=float)

        if N == 0:
            return np.ones_like(x), np.zeros_like(x)
        if N == 1:
            return x.copy(), np.ones_like(x)

        # P_{0}, P'_{0}
        Pn1 = np.ones_like(x)
        Dn1 = np.zeros_like(x)

        # P_{1}, P'_{1}
        Pn = x.copy()
        Dn = np.ones_like(x)

        for jj in range(2, N + 1):
            k = jj - 1  # current recurrence index, building P_{k+1}

            # Standard Legendre recurrence:
            # P_{k+1} = ((2k+1)x P_k - k P_{k-1}) / (k+1)
            P_temp = ((2 * k + 1) * x * Pn - k * Pn1) / (k + 1)

            # Derivative of the standard Legendre recurrence:
            # P'_{k+1} = ((2k+1)(P_k + x P'_k) - k P'_{k-1}) / (k+1)
            D_temp = ((2 * k + 1) * Pn + (2 * k + 1) * x * Dn - k * Dn1) / (k + 1)

            Pn1, Dn1 = Pn, Dn
            Pn, Dn = P_temp, D_temp

        return Pn, Dn

    if N == 0:
        return np.ones_like(x), np.zeros_like(x)

    Pn      = sp.special.eval_legendre(N, x)
    Pnm1    = sp.special.eval_legendre(N - 1, x)

    dPn     = N * (x * Pn - Pnm1) / (x**2 - 1.0)

    return Pn, dPn

# ---------------------------------------------------------------------
# Flipped Legendre-Radau polynomial
# ---------------------------------------------------------------------
def flipped_radau_polynomial(N: int, tau: np.ndarray, use_spartan: bool = USE_SPARTAN) -> np.ndarray:
    """Evaluate the flipped Legendre-Radau polynomial R_n(tau).

    Inputs:
        N :             Polynomial order
        tau :           Points to evaluate.
        use_spartan :   Legendre polynomimal computation method

    Outputs:        R_n(tau).

    """
    tau     = np.asarray(tau, dtype=float)

    Ln, _   = compute_legendre(N, tau, use_spartan=use_spartan)
    Lnm1, _ = compute_legendre(N - 1, tau, use_spartan=use_spartan)

    return Ln - Lnm1


def flipped_radau_polynomial_derivative(N: int, tau: np.ndarray) -> np.ndarray:
    """Derivative of the flipped Legendre-Radau polynomial."""
    tau         = np.asarray(tau, dtype=float)
    _, dLn      = compute_legendre(N, tau, use_spartan=USE_SPARTAN)
    _, dLnm1    = compute_legendre(N - 1, tau, use_spartan=USE_SPARTAN)

    return dLn - dLnm1


# ---------------------------------------------------------------------
# Compute flipped Radau nodes and quadrature weights
# ---------------------------------------------------------------------
# computes flipped Radau collocation nodes, full node set, quadrature weights
def flipped_radau_nodes_and_weights(N: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute flipped Radau collocation nodes, full node set, and quadrature weights.

    Inputs:
        N :     # of collocation nodes.

    Outputs:
        tau :   Collocation nodes (including +1, excluding -1).
        etau :  Full discrete node set for interpolation (includes -1).
        w :     Quadrature weights associated with tau-vector.

    """
    if N < 1:
        raise ValueError("N must be >= 1")

    # Degenerate one-node case.
    if N == 1:
        tau     = np.array([1.0])
        etau    = np.array([-1.0, 1.0])
        w       = np.array([2.0])
        return tau, etau, w

    # -------------------------------------------------------------
    # Compute LGR nodes with Newton-Raphson, flip to get fLGR.
    # -------------------------------------------------------------
    tau_std = -np.cos(2.0 * np.pi * np.arange(N) / (2 * (N - 1) + 1))
    tau_std = tau_std.astype(float)

    def radau_polynomial(x):
        Ln, _   = compute_legendre(N, np.array([x]), use_spartan=USE_SPARTAN)
        Lnm1, _ = compute_legendre(N - 1, np.array([x]), use_spartan=USE_SPARTAN)
        return float(Ln[0] + Lnm1[0])


    def radau_polynomial_derivative(x):
        _, dLn  = compute_legendre(N, np.array([x]), use_spartan=USE_SPARTAN)
        _, dLnm1= compute_legendre(N - 1, np.array([x]), use_spartan=USE_SPARTAN)
        return float(dLn[0] + dLnm1[0])

    # SPARTAN-style hardcoded Newton-Raphson iteration for LGR nodes
    if USE_HARDCODED_NEWTON:

        tau_old = np.ones_like(tau_std) * 2.0
        eps_tol = np.finfo(float).eps

        L       = np.zeros((N, N+1))
        idx     = np.arange(1, N)

        while np.max(np.abs(tau_std - tau_old)) > eps_tol:

            tau_old = tau_std.copy()

            # Construct Legendre Vandermonde matrix
            L[0, :] = (-1) ** np.arange(N+1)

            L[idx, 0] = 1
            L[idx, 1] = tau_std[idx]

            for k in range(2, N+1):
                L[idx, k] = ((2*k-1) * tau_std[idx] * L[idx, k-1] - (k-1) * L[idx, k-2]) / k

            tau_std[idx] = tau_old[idx] - ((1 - tau_old[idx]) / N) * \
                (L[idx, N-1] + L[idx, N]) / (L[idx, N-1] - L[idx, N])

    # SciPy-based Newton-Raphson root finding for LGR nodes
    else:

        for j in range(1, N):
            x0 = tau_std[j]
            tau_std[j] = sp.optimize.newton(
                radau_polynomial,
                x0,
                fprime=radau_polynomial_derivative,
                tol=1e-14,
                maxiter=100,
            )

    # extrapolate tau_std to get the full node set, then flip for fLGR
    tau     = np.sort(-tau_std)
    etau    = np.concatenate(([-1.0], tau))

    # -------------------------------------------------------------
    # Compute quadrature weights.
    # -------------------------------------------------------------
    weights_std = np.zeros(N)
    weights_std[0] = 2.0 / N**2
    for j in range(1, N):
        Ln, _ = compute_legendre(N - 1, np.array([tau_std[j]]), use_spartan=USE_SPARTAN)
        weights_std[j] = (1.0 - tau_std[j]) / (N * Ln[0]) ** 2

    w = np.flip(weights_std)

    return tau, etau, w


# ---------------------------------------------------------------------
# Differentiation matrix
# ---------------------full_nodes------------------------------------------------
# computes the differentiation matrix D, shape (N, N+1)

def differentiation_matrix(etau: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the pseudospectral differentiation matrix from the full node set.

    Inputs:  - 1D array of length m = N+1 containing the full node set (including -1).
    Outputs: D - differentiation matrix of shape (N, N+1) mapping state values at `full_nodes`
                 to derivatives at the collocation nodes (all nodes except the initial node).

    """
    xxPlusEnd = np.asarray(etau, dtype=float)
    M = len(xxPlusEnd)
    # M1 = M + 1
    # M2 = M * M

    # compute the barycentric weights
    Y       = np.tile(xxPlusEnd.reshape(-1, 1), (1, M))
    Ydiff   =  Y - Y.T + np.eye(M)

    WW      = np.tile((1.0 / np.prod(Ydiff, axis=1)).reshape(-1, 1), (1, M))
    D       = WW / (WW.T * Ydiff)

    # MATLAB: D(1:M1:M2) = 1-sum(D);
    np.fill_diagonal(D, 1.0 - np.sum(D, axis=0))

    # full differentiation matrix
    D       = -D.T
    D_full  = D.copy()

    # fLGR D-matrix
    D       = D[1:M, :]

    D2      = D @ D_full

    return D, D_full, D2

def differentiation_matrix_compare(full_nodes: np.ndarray) -> np.ndarray:
    """Compute the fLGR differentiation matrix via an alternative barycentric form."""
    x = np.asarray(full_nodes, dtype=float)
    m = len(x)

    # Pairwise differences: dX[i,j] = x_i - x_j
    dX = x[:, None] - x[None, :]

    # Barycentric weights:
    #   lambda_i = 1 / prod_{j != i} (x_i - x_j)
    dX_no_diag = dX + np.eye(m)
    lam = 1.0 / np.prod(dX_no_diag, axis=1)

    # Off-diagonal entries:
    #   D_ij = lambda_j / (lambda_i * (x_i - x_j)),  i != j
    D_full = np.outer(1.0 / lam, lam) / dX_no_diag

    # Fix diagonal entries so each row sums to zero
    np.fill_diagonal(D_full, 0.0)
    np.fill_diagonal(D_full, -np.sum(D_full, axis=1))

    # fLGR: remove the first row, keep all columns
    D = D_full[1:, :]

    return D


def flipped_radau_differential_operator(N: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the flipped Legendre-Radau nodes, weights, and differential operator.

    Inputs:
        N : Number of collocation nodes.

    Outputs:
        tau : Collocation nodes.
        etau : Full node set including -1.
        w : Quadrature weights.
        D : Differential operator mapping state values at etau to derivatives at collocation nodes tau.

    """
    tau, etau, w    = flipped_radau_nodes_and_weights(N)
    D, _, _         = differentiation_matrix(etau)

    #D_compare       = differentiation_matrix_compare(etau)
    #D = D_compare

    return tau, etau, w, D


# ---------------------------------------------------------------------
# hp composite differential operator
# ---------------------------------------------------------------------

def flipped_radau_hp_operator(N_col: int, H: int):
    """Compute the hp-composite fLGR nodes, weights, and per-interval D matrix.

    Instead of assembling a sparse global D matrix, returns the single local D block
    (scaled by H) along with the global node positions. The caller loops over intervals
    using D_local on the appropriate slice of the state vector.

    Inputs:
        N_col : Total number of collocation nodes (must be divisible by H).
        H     : Number of h-intervals.

    Outputs:
        tau_global  : Global collocation node positions in [-1, 1], shape (N_col,).
        etau_global : Full global node set (N_col + 1) including initial node at -1.
        w_global    : Quadrature weights mapped to global domain, shape (N_col,).
        D_local     : Single interval differentiation matrix scaled by H, shape (p, p+1).

    """
    if N_col % H != 0:
        raise ValueError(f"N_col={N_col} must be divisible by H={H}")

    p = N_col // H
    N_total = N_col + 1

    tau_local, etau_local, w_local, D_local = flipped_radau_differential_operator(p)

    etau_global = np.zeros(N_total)
    tau_global  = np.zeros(N_col)
    w_global    = np.zeros(N_col)

    for h in range(H):
        tau_start = -1.0 + 2.0 * h / H
        delta = 2.0 / H

        local_to_global = lambda tau_l, ts=tau_start, d=delta: ts + (tau_l + 1.0) * (d / 2.0)

        row_start = h * p
        col_start = h * p

        etau_global[col_start] = local_to_global(-1.0)
        for j in range(p):
            etau_global[col_start + 1 + j] = local_to_global(tau_local[j])
            tau_global[row_start + j] = local_to_global(tau_local[j])
            w_global[row_start + j] = w_local[j] * (delta / 2.0)

    return tau_global, etau_global, w_global, H * D_local


# ---------------------------------------------------------------------
# Lagrange interpolation (Eq. 9, CEAS2017)
# ---------------------------------------------------------------------

def lagrange_basis(eval_points: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """Compute Lagrange basis polynomials P_i(t) evaluated at eval_points."""
    t = np.asarray(eval_points, dtype=float)
    ti = np.asarray(nodes, dtype=float)

    m_eval = len(t)
    n_nodes = len(ti)

    P = np.ones((m_eval, n_nodes))

    for i in range(n_nodes):
        for k in range(n_nodes):
            if k != i:
                P[:, i] *= (t - ti[k]) / (ti[i] - ti[k])

    return P


def lagrange_interpolate(eval_points: np.ndarray, nodes: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Evaluate interpolating polynomial."""
    P = lagrange_basis(eval_points, nodes)
    return P @ values

