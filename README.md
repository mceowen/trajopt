# TrajOpt
TrajOpt is a self-contained Python library for multi-segment trajectory optimization using Sequential Convex Programming (SCP). The software is built to facilitate the design of __models__ that can effectively be re-used across different __mission__ scenarios.

## Features
* __Configurable Missions__: Once a model is built, configuring new missions is fast and efficient due to the config.yaml structure
* __Simple Model Design Framework__: Building new models is easy since you simply need to provide the continuous-time functions f_k(x, u, t, params, fcns).
* __Robust Base Algorithm__: The base AutoSCvx algorithm includes autotuning of penalty weights and second-order information for robust convergence on most problems with minimal hyper-parameter tuning.
* __Self Contained__: The base algorithm for solving the nonconvex problems is fully written in this package. The only black-boxes we accept are calls to the convex solvers due to their (mostly) gauranteed reliability and provable convergence.
* __Easy Algorithm Design__: Researchers can quickly design their own algorithms using the powerful modeling languages CVXPY and JAX, both of which closely resemble numpy.

## Segments
A trajectory __segment__ is defined by a set of (Costs, Constraints, Params, Fcns):

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
The __trajectory__ OCP is defined by summing the cost contributions and enforcing the constraints from each __segment__:

```math
\begin{aligned}
\underset{\{x_i,u_i,t_{I,i},t_{F,i}\}_{i=1}^{N}}{\mathrm{minimize}} 
\quad & \sum_{i=1}^{N} J_i \\
\mathrm{subject\;to} 
\quad & (z_1,\ldots,z_N) \in \mathbb{C}, \\
\mathrm{where} 
\quad \mathbb{C} 
&= \mathbb{C}_1 \times \mathbb{C}_2 \times \cdots \times \mathbb{C}_N
\end{aligned}
```

## Augmented Optimal Control Problem

## Trajectory Analyzer

## Method

## Sequential Convex Programming
