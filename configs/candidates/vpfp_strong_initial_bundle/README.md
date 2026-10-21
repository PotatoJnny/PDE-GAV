# Strong VPFP S0 for canonical Agent Iteration

`legacy_solver.py` is copied byte-for-byte from the previously tested strong
Solver (`c3fd9984dc05d1a8ecdb698e7c63861e98f201cadb0cfba9ad1b78d9e20867bd`).
Its empirical corrections are unchanged. Only the surrounding Candidate ABI
changes: `prepare_solution(case, solve_budget=None)` obtains public numerical
inputs and runs the complete solver before any scoring points are supplied.

This candidate explicitly chooses the historical 13 time, 33-by-33 space,
9-by-9 velocity discretization and matching public initial-macro preview.
Source, g0 and rho0 are queried on demand in batches of at most 8192 points.
These are candidate-owned mesh requirements, not a restriction imposed on
new Agent programs. Other candidates may choose any allowed public points.
The public Oracle is a borrowed SDK capability exposing only source, rho0
and g0, not a candidate-owned session. The Evaluator closes and revokes its
underlying session when the evaluation scope exits, including failed input
queries or preparation. The candidate must not call an unavailable `close`
method; it retains no Oracle reference in its returned solution handle.

The returned handle owns only four computed macro grids. It answers ordered,
duplicate-preserving POINTS using linear time / periodic bilinear spatial
interpolation over the whole torus, including the seam. This interpolation
adds numerical approximation; the old GRID solver is not misrepresented as
a native exact POINTS solver. The handle performs no PDE solve or Oracle
query after preparation and exposes no hidden truth.

At the fixed grid one preparation requests 1,146,717 source points, 88,209
g0 points and 1,089 rho0 points. Earlier identical-grid evaluations cost
roughly 17–20 seconds per case; a new subprocess/RPC run must measure its
actual cost. Solver-origin labels are stripped exactly as in that test.
