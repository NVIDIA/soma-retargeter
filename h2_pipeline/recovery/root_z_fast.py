"""Analytic/convex root-Z repair only; callers must run complete source admission.

Input heights must be independent FK of the ORIGINAL q at every120/50/600Hz
sample. A common interpolated root-Z translation adds exactly to both soles.
No joint, root XY, root rotation, frame count or timestamp is modified.
"""
import numpy as np
from scipy import sparse

FLOOR = 50e-6
CONTACT_MAX = 450e-6
DELTA_MIN, DELTA_MAX = -.05, .15
EXTRA_SPEED = .05
TOL = 1e-8


def solve_root_z(q_native, samples, fps=120):
    """Return (float64 q, report), or (None, report) without changing input.

    samples={rate:(times, original_heights[N,2], raw_grounded_mask[N,2])}.
    Every rate must contain its complete uniform grid over the native duration.
    Geometry/source-quality admission is deliberately outside this numerical step.
    """
    report = dict(passed=False, root_z_only=True, no_training_admission=True,
                  full_admission_performed=False, floor_safety_m=FLOOR,
                  grounded_safety_max_m=CONTACT_MAX, delta_bounds_m=[DELTA_MIN, DELTA_MAX],
                  extra_root_z_speed_cap_m_s=EXTRA_SPEED, constraint_tolerance_m=TOL)
    try:
        q = np.asarray(q_native, dtype=float)
        if fps != 120 or q.ndim != 2 or q.shape[1] != 38 or len(q) < 2 or not np.isfinite(q).all():
            raise ValueError('Expected finite native120Hz q with shape[N>=2,38]')
        n = len(q); duration = (n-1)/fps; records = {}; matrices = []; lowers = []; uppers = []
        for rate, values in samples.items():
            rate = int(rate)
            if rate in records or rate not in (120, 50, 600):
                raise ValueError('Require distinct120/50/600Hz samples')
            t, h, g = (np.asarray(v) for v in values)
            expected = np.arange(int(np.floor(duration*rate+1e-8))+1)/rate
            if (t.shape != expected.shape or not np.isfinite(t).all()
                    or not np.allclose(t, expected, rtol=0, atol=1e-10)
                    or h.shape != (len(t), 2) or not np.isfinite(h).all()
                    or g.shape != h.shape or not np.all((g == 0) | (g == 1))):
                raise ValueError('Invalid or incomplete independent sample grid: '+str(rate))
            t = t.astype(float); h = h.astype(float); g = g.astype(bool)
            left = np.minimum(np.floor(t*fps+1e-10).astype(int), n-1)
            alpha = np.clip(t*fps-left, 0., 1.); right = np.minimum(left+1, n-1)
            rows = np.arange(len(t))
            a = sparse.coo_matrix((np.r_[1-alpha, alpha],
                (np.r_[rows, rows], np.r_[left, right])), shape=(len(t), n)).tocsc()
            lo = FLOOR-h.min(axis=1)
            hi = np.where(g, CONTACT_MAX-h, np.inf).min(axis=1)
            matrices.append(a); lowers.append(lo); uppers.append(hi)
            records[rate] = (t, h, g, a)
        if set(records) != {120, 50, 600}:
            raise ValueError('All120/50/600Hz grids are required')
        lo = np.concatenate(lowers); hi = np.concatenate(uppers)
        if np.any(lo > hi) or np.any(lo > DELTA_MAX) or np.any(hi < DELTA_MIN):
            raise ValueError('Infeasible sole/contact intervals or root-Z trust bound')
        lower = max(float(lo.max()), DELTA_MIN); upper = min(float(hi.min()), DELTA_MAX)
        report.update(native_frames=n, fps=fps, constant_interval_m=[lower, upper],
                      sample_counts={str(r): len(v[0]) for r, v in records.items()})
        difference = sparse.diags([-np.ones(n-1), np.ones(n-1)], [0, 1], shape=(n-1, n), format='csc')
        if lower <= upper:
            delta = np.full(n, np.clip(0., lower, upper)); report['method'] = 'constant_offset'
        else:
            import osqp
            curvature = (sparse.diags([np.ones(n-2), -2*np.ones(n-2), np.ones(n-2)],
                         [0, 1, 2], shape=(n-2, n), format='csc') if n > 2 else sparse.csc_matrix((0, n)))
            eye = sparse.eye(n, format='csc')
            a = sparse.vstack([*matrices, eye, difference], format='csc')
            lower_all = np.r_[lo, np.full(n, DELTA_MIN), np.full(n-1, -EXTRA_SPEED/fps)]
            upper_all = np.r_[hi, np.full(n, DELTA_MAX), np.full(n-1, EXTRA_SPEED/fps)]
            # Solve in centimetres with order-one objective/constraint scales.
            # Global objective scaling preserves the same minimal/smooth delta.
            variable_scale = .01
            p = (eye+100*(curvature.T@curvature)).tocsc()
            solver = osqp.OSQP()
            solver.setup(P=p, q=np.zeros(n), A=a, l=lower_all/variable_scale, u=upper_all/variable_scale,
                         eps_abs=1e-7, eps_rel=1e-8, max_iter=5000,
                         polish=True, scaled_termination=False, adaptive_rho=False, rho=1., verbose=False)
            answer = solver.solve(); status = str(answer.info.status).lower()
            report.update(method='root_z_qp', qp_status=status, qp_iterations=int(answer.info.iter))
            if status not in ('solved', 'solved inaccurate') or answer.x is None or not np.isfinite(answer.x).all():
                raise ValueError('No valid finite QP solution: '+status)
            delta = np.asarray(answer.x, dtype=float)*variable_scale
        out = q.copy(); out[:, 2] += delta
        actual_delta = out[:, 2]-q[:, 2]  # Verify the coordinates actually returned.
        rates = {}; violations = [0., float(DELTA_MIN-actual_delta.min()), float(actual_delta.max()-DELTA_MAX)]
        extra_speed = float(np.max(np.abs(np.diff(actual_delta)))*fps)
        violations.append(float((extra_speed-EXTRA_SPEED)/fps))
        for rate, (t, h, g, a) in records.items():
            after = h+(a@actual_delta)[:, None]
            penetration = max(0., float(FLOOR-after.min()))
            contact = max(0., float(after[g].max()-CONTACT_MAX)) if g.any() else 0.
            rates[str(rate)] = dict(samples=len(t), original_min_height_m=float(h.min()),
                min_height_m=float(after.min()), max_height_m=float(after.max()),
                grounded_samples=int(g.sum()), max_grounded_height_m=float(after[g].max()) if g.any() else None,
                floor_constraint_violation_m=penetration, contact_constraint_violation_m=contact)
            violations.extend([penetration, contact])
        fixed = np.r_[0:2, 3:38]
        exact = np.array_equal(out[:, fixed], q[:, fixed])
        maximum = max(violations)
        report.update(rates=rates, max_constraint_violation_m=maximum,
            delta_min_m=float(actual_delta.min()), delta_max_m=float(actual_delta.max()),
            max_extra_root_z_speed_m_s=extra_speed, fixed_root_xy_rotation_joints_exact=exact,
            delta_bound_violation_m=max(0., float(DELTA_MIN-actual_delta.min()), float(actual_delta.max()-DELTA_MAX)),
            extra_root_z_speed_violation_m_s=max(0., extra_speed-EXTRA_SPEED),
            timebase_changed=False, joint_order_changed=False)
        if not np.isfinite(out).all() or not exact or maximum > TOL:
            raise ValueError('Independent returned-coordinate constraint verification failed')
        report['passed'] = True
        return out, report
    except (ValueError, TypeError, KeyError, ImportError) as exc:
        report['reason'] = type(exc).__name__+': '+str(exc)
        return None, report


solve = solve_root_z


if __name__ == '__main__':
    import json
    q = np.zeros((121, 38)); q[:, 2] = .6; q[:, 3] = 1.; q[:, 7:] = np.arange(31)*.001
    def grids(function):
        result = {}
        for rate in (120, 50, 600):
            t = np.arange(rate+1)/rate
            h = function(t)
            result[rate] = (t, h, np.ones(h.shape, dtype=bool))
        return result
    constant = grids(lambda t: np.c_[np.full(len(t), -.004), np.full(len(t), -.0039)])
    out, report = solve(q, constant)
    assert out is not None and report['method'] == 'constant_offset'
    assert abs(report['delta_min_m']-.00405) < 1e-12
    varying = grids(lambda t: np.c_[-.004-.0015*np.sin(2*np.pi*t), -.00395-.0015*np.sin(2*np.pi*t)])
    out, report = solve(q, varying)
    assert out is not None and report['method'] == 'root_z_qp', report
    assert report['max_constraint_violation_m'] <= TOL and report['max_extra_root_z_speed_m_s'] <= EXTRA_SPEED+120*TOL
    impossible = grids(lambda t: np.c_[np.full(len(t), -.002), np.full(len(t), .002)])
    out, bad = solve(q, impossible)
    assert out is None and not bad['passed']
    assert np.all(q[:, 2] == .6) and np.all(q[:, 3] == 1.)
    print(json.dumps(dict(constant_offset=True, varying_feasible_qp=True, incompatible_contact_rejected=True,
                         original_input_unchanged=True, varying_report=report), indent=2))
