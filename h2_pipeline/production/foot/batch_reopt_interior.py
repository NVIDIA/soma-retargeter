"""Unchanged retarget pipeline with validated interior QPs and OSQP fallback."""
from pathlib import Path
import json
import sys
import time
import batch_reopt as B
import batch_reopt_gpu as D
import batch_reopt_hybrid as H
import batch_reopt_hybrid_parallel as M
import interior_qp

ORIGINAL_INITIALIZE = M.initialize
ORIGINAL_RUN = H.run
CONTEXT = None


def initialize(devices):
    global CONTEXT
    sys.path.insert(0, str(B.HERE/'vendor_piqp'))
    ORIGINAL_INITIALIZE(devices)
    CONTEXT = interior_qp.Context(H.QP.original)
    H.QP.original = CONTEXT


def run(row, config):
    CONTEXT.reset()
    retained_original = 'torso' in row['category'].split('+')
    begin = time.perf_counter()
    qp_calls, qp_seconds = H.QP.calls, H.QP.seconds
    geometry_calls, geometry_seconds = D.G.calls, D.G.seconds
    if retained_original:
        H.QP.original = CONTEXT.original
    try:
        result = ORIGINAL_RUN(row, config)
    finally:
        H.QP.original = CONTEXT
    first_attempt = None
    if result['status'] not in ('legacy_accepted', 'new_H2_reference_validated') and CONTEXT.fast:
        first_attempt = dict(status=result['status'], reason=result.get('reason'),
                             seconds=result.get('seconds'))
        H.QP.original = CONTEXT.original
        try:
            result = ORIGINAL_RUN(row, config)
        finally:
            H.QP.original = CONTEXT
    result['seconds'] = round(time.perf_counter()-begin, 2)
    backend = result['execution_backend']
    backend.update(name='shared_cuda_geometry_piqp_with_original_osqp_fallback',
                   qp_calls=H.QP.calls-qp_calls,
                   qp_wait_and_solve_seconds=H.QP.seconds-qp_seconds,
                   upper_gpu_evaluations=D.G.calls-geometry_calls,
                   upper_gpu_evaluation_seconds=D.G.seconds-geometry_seconds,
                   interior_qps=CONTEXT.fast, original_osqp_fallback_qps=CONTEXT.fallback,
                   interior_attempt_seconds=CONTEXT.attempt_seconds,
                   fallback_reasons=dict(CONTEXT.reasons),
                   retained_original_motion=retained_original,
                   original_motion_retry=first_attempt)
    if result.get('report'):
        report = B.read(result['report'])
        report['batch_validation']['execution_backend'] = backend
        report['batch_validation']['seconds'] = result['seconds']
        B.save(Path(result['report']), report)
        result['report_sha256'] = B.sha(result['report'])
    return result


def main():
    if '--dry-run' in sys.argv or '--help' in sys.argv:
        M.main(); return
    out = Path(sys.argv[sys.argv.index('--out')+1]).resolve()
    vendor = B.HERE/'vendor_piqp'
    libraries = sorted(vendor.rglob('*.so'))
    if not libraries:
        raise RuntimeError('Validated PIQP 0.6.4 vendor package is required')
    manifest = dict(schema='original_qp_interior_fallback_v3', piqp_version='0.6.4',
                    original_problem_and_admission=True,
                    candidate_checks='Original unscaled primal/dual tolerances; original final CPU admission',
                    fallback='Unmodified original CPU OSQP setup/settings',
                    torso_motions='Complete original OSQP pipeline',
                    unsuccessful_motion='One complete original OSQP pipeline retry after interior QPs',
                    code={str(p): B.sha(p) for p in [Path(__file__), B.HERE/'interior_qp.py',
                        B.HERE/'batch_reopt_hybrid.py', B.HERE/'batch_reopt_hybrid_parallel.py',
                        B.HERE/'shared_gpu_geometry.py', B.HERE/'upper_feasibility.py',
                        B.HERE/'reconstruct_bvh_feet_v14.py', *libraries]})
    out.mkdir(parents=True, exist_ok=True)
    receipt = out/'interior_qp_manifest.json'
    if receipt.exists() and B.read(receipt) != manifest:
        raise ValueError('Interior QP implementation changed')
    if not receipt.exists():
        B.save(receipt, manifest)
    M.initialize = initialize
    H.run = run
    M.main()


if __name__ == '__main__':
    main()
