"""Exercise the recorded fast QP and deliberate original-solver fallback."""
import importlib.util
import numpy as np
import pytest
from scipy import sparse
from h2_pipeline.workspace import PRODUCTION


def test_fast_qp_and_preserved_torso_size_match_original_solution():
    osqp = pytest.importorskip('osqp')
    pytest.importorskip('piqp')
    path = PRODUCTION / 'foot/interior_qp.py'
    spec = importlib.util.spec_from_file_location('production_interior_qp', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    context = module.Context(osqp.OSQP)
    P = sparse.eye(2, format='csc'); A = sparse.csc_matrix([[1., 0.], [0., -1.], [1., 1.]])
    q = np.array([-.2, .1]); lower = np.array([-1., -1., -1.]); upper = np.array([1., 1., 1.])
    def solve():
        solver = context(); solver.setup(P=P, q=q, A=A, l=lower, u=upper,
                                         eps_abs=1e-8, eps_rel=1e-8, max_iter=30000, polishing=True, verbose=False)
        return solver.solve().x
    fast = solve()
    np.testing.assert_allclose(fast, [.2, -.1], atol=1e-7)
    assert context.fast == 1 and context.fallback == 0
    context.original_sizes.add(2)
    original = solve()
    np.testing.assert_allclose(original, fast, atol=1e-7)
    assert context.fallback == 1
