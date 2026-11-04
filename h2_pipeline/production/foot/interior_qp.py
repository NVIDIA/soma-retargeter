"""Solve the exact original QP with PIQP; retain original OSQP as fallback.

Only exact +/-identity rows become variable bounds. Every returned PIQP
candidate is checked against the original unsplit matrices and tolerances.
"""
from collections import Counter
from types import SimpleNamespace
import time
import numpy as np
from scipy import sparse


class Context:
    def __init__(self, original_osqp, attempt_max_iter=80):
        import piqp
        self.piqp = piqp
        self.original = original_osqp
        self.attempt_max_iter = attempt_max_iter
        self.reset()

    def reset(self):
        self.fast = self.fallback = 0
        self.attempt_seconds = 0.
        self.reasons = Counter()
        self.original_sizes = set()

    def __call__(self, algebra='builtin'):
        if algebra != 'builtin':
            raise ValueError('Interior QP adapter requires the original CPU fallback')
        return InteriorQP(self)


class InteriorQP:
    def __init__(self, context):
        self.context = context

    def setup(self, P, q, A, l, u, **settings):
        self.P = sparse.triu(P, format='csc')
        self.q = np.asarray(q, dtype=float)
        self.A = A.tocsc()
        self.l, self.u = np.asarray(l, dtype=float), np.asarray(u, dtype=float)
        self.settings = dict(settings)

    def original_solve(self, reason, **kwargs):
        self.context.fallback += 1
        self.context.reasons[reason] += 1
        solver = self.context.original(algebra='builtin')
        solver.setup(P=self.P, q=self.q, A=self.A, l=self.l, u=self.u, **self.settings)
        return solver.solve(**kwargs)

    def solve(self, **kwargs):
        if self.P.shape[0] in self.context.original_sizes:
            return self.original_solve('preserved_torso_target_stage', **kwargs)
        begin = time.perf_counter()
        try:
            answer = self.solve_interior()
        except Exception as error:
            self.context.attempt_seconds += time.perf_counter()-begin
            return self.original_solve(type(error).__name__+': '+str(error)[:120], **kwargs)
        self.context.attempt_seconds += time.perf_counter()-begin
        self.context.fast += 1
        return answer

    def solve_interior(self):
        begin = time.perf_counter()
        P, A, q, l, u = self.P, self.A, self.q, self.l, self.u
        n, m = len(q), len(l)
        eq = np.isfinite(l) & np.isfinite(u) & (l == u)
        csr = A.tocsr()
        single = np.diff(csr.indptr) == 1
        values, columns = np.zeros(m), np.zeros(m, dtype=int)
        rows = np.flatnonzero(single)
        values[rows] = csr.data[csr.indptr[rows]]
        columns[rows] = csr.indices[csr.indptr[rows]]
        box = single & (abs(values) == 1) & ~eq
        rows = np.flatnonzero(box)
        coefficients, cols = values[rows], columns[rows]
        lower = np.where(coefficients > 0, l[rows], -u[rows])
        upper = np.where(coefficients > 0, u[rows], -l[rows])
        xl, xu = np.full(n, -np.inf), np.full(n, np.inf)
        np.maximum.at(xl, cols, lower)
        np.minimum.at(xu, cols, upper)
        if np.any(xl > xu):
            raise ValueError('Inconsistent variable bounds')
        general = ~eq & ~box & (np.isfinite(l) | np.isfinite(u))
        fullP = P+P.T-sparse.diags(P.diagonal())
        solver = self.context.piqp.SparseSolver()
        settings = solver.settings
        eps_abs = self.settings.get('eps_abs', 1e-3)
        eps_rel = self.settings.get('eps_rel', 1e-3)
        settings.eps_abs = eps_abs
        settings.eps_rel = eps_rel
        settings.eps_duality_gap_abs = eps_abs
        settings.eps_duality_gap_rel = eps_rel
        settings.max_iter = self.context.attempt_max_iter
        settings.verbose = False
        settings.compute_timings = True
        solver.setup(fullP, q,
            A=A[eq].tocsc() if eq.any() else None, b=u[eq] if eq.any() else None,
            G=A[general].tocsc() if general.any() else None,
            h_l=l[general] if general.any() else None,
            h_u=u[general] if general.any() else None, x_l=xl, x_u=xu)
        status = solver.solve()
        if status != self.context.piqp.Status.PIQP_SOLVED:
            raise RuntimeError(str(status))
        result = solver.result
        x = np.asarray(result.x).copy()
        y = np.zeros(m)
        y[eq] = result.y
        y[general] = np.asarray(result.z_u)-np.asarray(result.z_l)
        # Put each aggregated box multiplier back on an original tight row.
        # Tied duplicate bounds need only one owner for valid stationarity.
        lo_owner, hi_owner = np.full(n, m, dtype=int), np.full(n, m, dtype=int)
        matches = np.isfinite(lower) & (lower == xl[cols])
        np.minimum.at(lo_owner, cols[matches], rows[matches])
        matches = np.isfinite(upper) & (upper == xu[cols])
        np.minimum.at(hi_owner, cols[matches], rows[matches])
        valid = lo_owner < m
        np.add.at(y, lo_owner[valid], -np.asarray(result.z_bl)[valid]/values[lo_owner[valid]])
        valid = hi_owner < m
        np.add.at(y, hi_owner[valid], np.asarray(result.z_bu)[valid]/values[hi_owner[valid]])
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('Nonfinite interior QP candidate')
        ax = A@x
        projected = np.clip(ax, l, u)
        primal = float(np.max(abs(ax-projected), initial=0.))
        px, aty = fullP@x, A.T@y
        dual = float(np.max(abs(px+q+aty), initial=0.))
        norm = lambda value: float(np.max(abs(value), initial=0.))
        primal_limit = eps_abs+eps_rel*max(norm(ax), norm(projected))
        dual_limit = eps_abs+eps_rel*max(norm(px), norm(q), norm(aty))
        if primal > primal_limit or dual > dual_limit:
            raise ValueError('Original QP residual tolerances not met')
        # PIQP also checks the objective gap using the original requested
        # absolute and relative tolerances before returning PIQP_SOLVED.
        elapsed = time.perf_counter()-begin
        info = SimpleNamespace(status='solved', status_val=1, iter=int(result.info.iter),
                               prim_res=primal, dual_res=dual, run_time=elapsed,
                               setup_time=float(result.info.setup_time),
                               solve_time=float(result.info.solve_time),
                               obj_val=float(.5*x@px+q@x), qp_backend='piqp')
        return SimpleNamespace(x=x, y=y, info=info)
