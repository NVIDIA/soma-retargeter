"""Recognize a converged initial fixed-reference fit before constructing a QP."""
import math


def install(batch):
    original=batch.U.Problem.reconstruct
    def reconstruct(problem,iterations):
        if hasattr(problem,'fixed_goal'):
            score,_=problem.source_measure(problem.x0)
            # The objective is a sum of squares with lower bound zero. The
            # original fixed-goal loop only accepts improvements >1e-10.
            if (score['physical'] and math.isfinite(score['cost'])
                    and 0<=score['cost']<=1e-14):
                return problem.x0.copy(),dict(method='initial_fixed_goal_already_converged',
                    before=score,after=score,history=[],iterations=0,optimization_performed=False,
                    nonnegative_objective_lower_bound=0.,stopping_cost=1e-14,
                    original_minimum_accepted_improvement=1e-10,
                    fixed_reference_sha256=batch.sha(problem.fixed_goal['path']),
                    source_goals='Unchanged pre-frozen H2 reference; complete final fidelity and source checks still required')
        return original(problem,iterations)
    batch.U.Problem.reconstruct=reconstruct
    return dict(format='fixed_goal_initial_convergence_v1',admission_changed=False,
                applies_only='Physically feasible fixed-reference seed with nonnegative objective <=1e-14')
