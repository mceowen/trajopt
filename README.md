# TrajOpt
TrajOpt is a self-contained Python library for multi-segment trajectory optimization using Sequential Convex Programming (SCP).

## Segments

```math
\begin{aligned}
J_i &= \sum_j \int_{t_{I,i}}^{t_{F,i}} 
J_{r,j}(x_i,u_i,t,\mathrm{params}_i,\mathrm{fcns}_i)\,dt 
+ \sum_j J_{F,j}\!\left(
x_i(t_{F,i}),u_i(t_{F,i}),t_{F,i},
\mathrm{params}_i,\mathrm{fcns}_i
\right) \\
\mathbb{C}_i 
&= \left\{
(x_i(\cdot),u_i(\cdot),t_{I,i},t_{F,i})
\;\middle|\;
\begin{aligned}
x_i(t_{I,i}) &= x_{I,i}, \\
x_i(t_{F,i}) &= x_{F,i}, \\
\dot{x}_i &= f_i(x_i,u_i,t_i,\mathrm{params}_i,\mathrm{fcns}_i), \\
lb_{g,i} &\le g_i(x_i,u_i,t_i,\mathrm{params}_i,\mathrm{fcns}_i) \le ub_{g,i}, \\
lb_{g_{\mathrm{cvx}},i} &\le g_{\mathrm{cvx},i}(x_i,u_i,t_i,\mathrm{params}_i,\mathrm{fcns}_i) 
\le ub_{g_{\mathrm{cvx}},i}
\end{aligned}
\right\}. \\
\mathrm{params}_i 
&= \text{a set of user-defined variables} \\
\mathrm{fcns}_i 
&= \text{a set of user-defined functions of the form } 
f_k(x_i,u_i,t_i,\mathrm{params}_i,\mathrm{fcns}_i)
\end{aligned}
```

## Trajectory

## Augmented Optimal Control Problem

## Trajectory Analyzer

## Method

## Sequential Convex Programming
