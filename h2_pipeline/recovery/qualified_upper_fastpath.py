"""Reuse an immutable, already-qualified upper reference; do not change admission."""
from pathlib import Path
import numpy as np


def upper_pass(report):
    checks=[c for c in report['checks'] if c['section'] in ('arms','torso')]
    return bool(checks) and all(c['passed'] for c in checks)


def install(batch):
    original=batch.U.Problem.anatomical_project
    def project(problem,iterations,wrist_budget=15.,checkpoint=None):
        source=batch.SOURCE/'refined/csv'/f'{problem.key}.csv'
        report_path=batch.SOURCE/'reports'/f'{problem.key}.json'
        source_report=batch.read(report_path)
        qualified=(upper_pass(source_report) and upper_pass(problem.seed_report)
                   and problem.seed_report['physical_and_margin_pass'])
        if qualified:
            source_sha=batch.sha(source)
            assert source_sha==source_report['refined_sha256'], 'Qualified source CSV changed'
            assert source_report==problem.original, 'Original source report changed'
            assert batch.API.raw_valid(Path(problem.job['work']),problem.job), 'Raw BVH/G1 provenance changed'
            header,raw,cols,q=batch.U.load(source,problem)
            protected=np.r_[3:7,[7+i for i,name in enumerate(problem.names)
                                   if not any(s in name for s in ('hip','knee','ankle'))]]
            exact=(q.shape==problem.q0.shape and header==problem.header
                   and np.array_equal(raw[:,0],problem.raw[:,0])
                   and np.array_equal(q[:,protected],problem.q0[:,protected]))
            if exact:
                return problem.x0.copy(),dict(method='reuse_immutable_qualified_source_upper',
                    source_csv=str(source),source_csv_sha256=source_sha,
                    source_report=str(report_path),source_report_sha256=batch.sha(report_path),
                    source_and_repaired_seed_all_upper_g1_pass=True,seed_physical_pass=True,
                    root_rotation_upper_waist_head_exact=True,upper_optimization_performed=False,
                    raw_bvh_g1_pair_verified=True,
                    remaining_checks='Unchanged independent goal audit, frozen reference, distinct refit and complete final admission')
        return original(problem,iterations,wrist_budget,checkpoint)
    batch.U.Problem.anatomical_project=project
    return dict(format='qualified_original_upper_optimizer_early_exit_v1',admission_changed=False,
                otherwise='Original anatomical optimizer with bounded iteration/time budget')
