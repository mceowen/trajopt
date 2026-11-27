"""Replay a converged GHAME solution with its native control interpolant and check it against the problem limits."""
import numpy as np
from scipy.integrate import solve_ivp

RTOL, ATOL = 1e-12, 1e-14
# intersample allowance on the path limits and state bounds, which are imposed only at the nodes
PATH_ALLOW = 0.02


def mesh_nodes(N, segments, kind, knots=None):
    """Normalized mesh of a transcription: uniform for shooting, flipped Radau per segment for collocation."""
    if kind == "ms":
        return np.linspace(0.0, 1.0, N)
    from trajopt.methods.common.discretize import flipped_radau_hp_operator
    _, nodes, _, _ = flipped_radau_hp_operator(N - 1, segments, knots=None if knots is None else 2.0 * np.asarray(knots, float) - 1.0)
    tau = (np.asarray(nodes) + 1.0) / 2.0
    tau[0], tau[-1] = 0.0, 1.0
    return tau


def control_interpolant(sigma, u, kind, segments=1, control_interpolation="lagrange", stride=1):
    """Return u(s) for s in [0, 1] following the interpolation the transcription optimized with."""
    sigma, u = np.asarray(sigma, float), np.asarray(u, float)
    N = len(sigma)
    if kind == "ms" or control_interpolation == "foh":
        knots = np.arange(0, N, stride) if kind != "ms" else np.arange(N)
        if knots[-1] != N - 1:
            knots = np.append(knots, N - 1)
        sk, uk = sigma[knots], u[knots]
        return lambda s: np.array([np.interp(s, sk, uk[:, j]) for j in range(u.shape[1])])
    d = (N - 1) // segments
    def f(s):
        h = min(int(np.searchsorted(sigma[d::d], s, side="left")), segments - 1)
        idx = np.arange(h * d + 1, (h + 1) * d + 1)
        x, y = sigma[idx], u[idx]
        w = np.array([1.0 / np.prod([x[i] - x[j] for j in range(d) if j != i]) for i in range(d)])
        diff = s - x
        if np.any(np.abs(diff) < 1e-14):
            return y[np.argmin(np.abs(diff))]
        b = w / diff
        return (b @ y) / b.sum()
    return f


def dilation_interpolant(sigma, t, kind, segments=1):
    """Piecewise-constant dilation s = dt/dsigma recovered from the node times."""
    sigma, t = np.asarray(sigma, float), np.asarray(t, float)
    N = len(sigma)
    if kind == "ms":
        edges = sigma
    else:
        d = (N - 1) // segments
        edges = sigma[::d]
    te = np.interp(edges, sigma, t)
    rates = np.diff(te) / np.diff(edges)
    return lambda s: rates[min(int(np.searchsorted(edges[1:], s, side="right")), len(rates) - 1)]


def replay(run, limits, scales, kind, segments=1, control_interpolation="lagrange", stride=1, samples=64, knots=None):
    """Propagate the optimized control with DOP853 and evaluate the acceptance checks of the comparison."""
    fin = next(iter(run.evaluation.phases.values())).final
    t, x, u = np.asarray(fin.t_opt, float), np.asarray(fin.x_opt, float), np.asarray(fin.u_opt, float)
    N = len(t)
    sigma = mesh_nodes(N, segments, kind, knots)
    u_of = control_interpolant(sigma, u, kind, segments, control_interpolation, stride)
    s_of = dilation_interpolant(sigma, t, kind, segments)
    dyn, params, fcns = limits["dynamics"], limits["params"], limits["fcns"]

    def rhs(s, y):
        xx, tt = y[:6], y[6]
        uu = u_of(s)
        f = np.asarray(dyn(xx, uu, tt, params, fcns), float)
        return s_of(s) * np.append(f, 1.0)

    y0 = np.append(x[0], t[0])
    sol = solve_ivp(rhs, (0.0, 1.0), y0, method="DOP853", rtol=RTOL, atol=ATOL, dense_output=True)
    ss = np.unique(np.concatenate([np.linspace(sigma[k], sigma[k + 1], samples + 1) for k in range(N - 1)]))
    Y = sol.sol(ss).T
    X, T = Y[:, :6], Y[:, 6]
    U = np.array([u_of(s) for s in ss])
    S = np.array([s_of(s) for s in ss])
    path = {name: np.array([float(np.asarray(fn(X[i], U[i], T[i], params, fcns)).ravel()[0]) for i in range(len(ss))])
            for name, fn in limits["path_fcns"].items()}
    checks = {}
    # terminal prescribed components (scaled units)
    for name, (j, target, scale) in limits["terminal"].items():
        checks[f"terminal {name}"] = (abs(X[-1, j] - target) / scale, 0.001)
    # path limits and state bounds: PATH_ALLOW of the bound; control bounds and rates: 0.1 percent
    for name, (lim, arr) in {k: (limits["path"][k], path[k]) for k in path}.items():
        checks[name] = (max(0.0, float(np.max(arr) - lim)), PATH_ALLOW * abs(lim))
    for j, (lo, hi) in enumerate(limits["state_bounds"]):
        viol = max(0.0, float(np.max(X[:, j] - hi)), float(np.max(lo - X[:, j])))
        checks[f"state {limits['state_names'][j]}"] = (viol, PATH_ALLOW * max(abs(lo), abs(hi)))
    for j, (lo, hi) in enumerate(limits["control_bounds"]):
        viol = max(0.0, float(np.max(U[:, j] - hi)), float(np.max(lo - U[:, j])))
        checks[f"control {limits['control_names'][j]}"] = (viol, 1e-3 * max(abs(lo), abs(hi)))
    dU = np.gradient(U, ss, axis=0) / S[:, None]
    for j, lim in enumerate(limits["control_rate"]):
        checks[f"rate {limits['control_names'][j]}"] = (max(0.0, float(np.max(np.abs(dU[:, j])) - lim)), 1e-3 * lim)
    # local dynamics: propagate each interval from its node and compare with the next node (scaled)
    local = 0.0
    for k in range(N - 1):
        yk = np.append(x[k], t[k])
        sk = solve_ivp(rhs, (sigma[k], sigma[k + 1]), yk, method="DOP853", rtol=RTOL, atol=ATOL)
        local = max(local, float(np.max(np.abs((sk.y[:6, -1] - x[k + 1]) / scales))))
    checks["dynamics"] = (local, 0.001)
    checks["endpoint drift"] = (float(np.max(np.abs((X[-1] - x[-1]) / scales))), 0.001)
    checks["duration"] = (max(0.0, -float(T[-1] - T[0])), 1e-9)
    ratios = {k: v / a for k, (v, a) in checks.items()}
    limiting = max(ratios, key=ratios.get)
    return dict(R=float(ratios[limiting]), limiting=limiting, checks={k: dict(v=float(v), a=float(a)) for k, (v, a) in checks.items()},
                cost=float(X[-1, 2]), nodal_cost=float(x[-1, 2]), t_final=float(T[-1]),
                dense=dict(sigma=ss, t=T, x=X, u=U, path=path), nodes=dict(sigma=sigma, t=t, x=x, u=u))
