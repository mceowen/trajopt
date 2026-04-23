import casadi as ca
import numpy as np


def _rk4(f, x, u, t, dt, params, fcns):
    k1 = f(t,        x,              u, params, fcns)
    k2 = f(t + dt/2, x + (dt/2)*k1, u, params, fcns)
    k3 = f(t + dt/2, x + (dt/2)*k2, u, params, fcns)
    k4 = f(t + dt,   x + dt*k3,     u, params, fcns)
    return x + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)


def solve(ta):
    cfg     = ta.config
    problem = ta.problem
    N       = cfg.method.time_grid
    nx      = problem.index_map.n.state
    nu      = problem.index_map.n.control
    params  = problem.params
    fcns    = problem.fcns
    dyn     = fcns.dynamics

    w_sym, lbw, ubw = [], [], []
    g_sym, lbg, ubg = [], [], []

    Xs = []
    for k in range(N):
        Xk = ca.MX.sym(f"x{k}", nx)
        Xs.append(Xk)
        w_sym.append(Xk)
        lbw.extend([-ca.inf] * nx)
        ubw.extend([ ca.inf] * nx)

    Us = []
    for k in range(N - 1):
        Uk = ca.MX.sym(f"u{k}", nu)
        Us.append(Uk)
        w_sym.append(Uk)
        lbw.extend([-ca.inf] * nu)
        ubw.extend([ ca.inf] * nu)

    tf_sym = ca.MX.sym("tf")
    w_sym.append(tf_sym)
    lbw.append(float(cfg.method.bounds.tf_lb))
    ubw.append(float(cfg.method.bounds.tf_ub))

    dt = tf_sym / (N - 1)

    for k in range(N - 1):
        tk     = k * dt
        x_next = _rk4(dyn, Xs[k], Us[k], tk, dt, params, fcns)
        g_sym.append(Xs[k + 1] - x_next)
        lbg.extend([0.0] * nx)
        ubg.extend([0.0] * nx)

    for cnstr in problem.constraints.get(type='equality_bc'):
        node = Xs[cnstr.boundary_idx]
        val  = np.atleast_1d(cnstr.value)
        for i, v in zip(cnstr.idx, val):
            g_sym.append(node[int(i)] - float(v))
            lbg.append(0.0)
            ubg.append(0.0)

    for cnstr in problem.constraints.get(type='nonconvex_inequality'):
        raw_dim = cnstr.dimension // 2 if cnstr.upper_and_lower else cnstr.dimension
        for k in cnstr.nodes:
            k   = int(k)
            tk  = k * dt
            xk  = Xs[k]
            uk  = Us[k] if k < N - 1 else Us[-1]
            g_k = cnstr.fcn_dim(tk, xk, uk, params)

            if cnstr.min_value is not None and cnstr.max_value is not None:
                for i in range(raw_dim):
                    g_sym.append(g_k[i])
                    lbg.append(float(cnstr.min_value[i]))
                    ubg.append(float(cnstr.max_value[i]))
            elif cnstr.max_value is not None:
                for i in range(raw_dim):
                    g_sym.append(g_k[i])
                    lbg.append(-ca.inf)
                    ubg.append(float(cnstr.max_value[i]))
            elif cnstr.min_value is not None:
                for i in range(raw_dim):
                    g_sym.append(g_k[i])
                    lbg.append(float(cnstr.min_value[i]))
                    ubg.append(ca.inf)

    obj = ca.MX(0)
    for cost in problem.costs.costs_list:
        if cost.type == 'terminal_state':
            sign = getattr(cost, 'sign', 1)
            for i in cost.idx:
                obj = obj + sign * Xs[-1][int(i)]
        elif cost.type == 'min_time':
            obj = obj + tf_sym

    nlp        = {"x": ca.vertcat(*w_sym), "f": obj, "g": ca.vertcat(*g_sym)}
    solver_cfg = cfg.method.get('solver', {})
    ipopt_opts = {k: v for k, v in solver_cfg.items() if k != 'name'}
    solver     = ca.nlpsol("ms", "ipopt", nlp, {"ipopt": ipopt_opts})

    guess   = cfg.method.guess
    x_start = np.array(guess.x_start)
    x_end   = np.array(guess.x_end)
    u_g     = np.array(guess.u)
    tf_g    = float(guess.tf)

    X_g = np.array([x_start + (x_end - x_start) * k / (N - 1) for k in range(N)])
    U_g = np.tile(u_g, (N - 1, 1))
    w0  = np.concatenate([X_g.flatten(), U_g.flatten(), [tf_g]])

    sol   = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    w_val = sol["x"].full().flatten()

    X_sol  = np.array([w_val[k*nx:(k+1)*nx] for k in range(N)])
    U_sol  = np.array([w_val[N*nx + k*nu:N*nx + (k+1)*nu] for k in range(N - 1)])
    tf_sol = float(w_val[-1])

    return {
        "t":   np.linspace(0.0, tf_sol, N),
        "x":   X_sol,
        "u":   U_sol,
        "tf":  tf_sol,
        "obj": float(sol["f"]),
        "raw": sol,
    }
