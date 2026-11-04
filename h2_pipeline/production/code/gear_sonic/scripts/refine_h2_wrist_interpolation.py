"""Fit native wrist knots to native and downstream resampled orientation targets.

Source timestamps, body/limb goals, and every non-wrist CSV column are preserved.
This addresses FK interpolation error; it does not filter the source motion.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R, Slerp
from gear_sonic.scripts.refine_h2_bvh_arm_targets import arm_joint_bounds


def refine(model, q, targets, side, native_fps, output_fps, limit_deg=.18, joint_limit_margin_deg=2.):
    m = mujoco.MjModel.from_xml_path(str(model))
    d = mujoco.MjData(m)
    js = np.array([i for i in range(1, m.njnt)
                   if m.joint(i).name.startswith(side.lower() + '_wrist_')])
    qi = m.jnt_qposadr[js]
    knot_lo, knot_hi = arm_joint_bounds(m, js, joint_limit_margin_deg)
    if np.any(q[:, qi] < knot_lo - 1e-6) or np.any(q[:, qi] > knot_hi + 1e-6):
        raise ValueError('Input wrists violate the requested margin; rerun the bounded arm IK first')
    body = m.body(side.lower() + '_wrist_yaw_link').id
    native_times = np.arange(len(q)) / native_fps
    times = np.arange(int(np.floor(native_times[-1] * output_fps + 1e-8)) + 1) / output_fps
    goals = Slerp(native_times, targets)(times)
    root = Slerp(native_times, R.from_quat(q[:, [4, 5, 6, 3]]))(times)
    result = q.copy()
    repaired = []
    def error(pose, target):
        d.qpos[:] = pose
        mujoco.mj_kinematics(m, d)
        return (target.inv() * R.from_matrix(d.xmat[body].reshape(3, 3))).as_rotvec()
    for k, t in enumerate(times):
        index = t * native_fps
        a = min(int(np.floor(index + 1e-9)), len(q) - 1)
        alpha = max(0., index - a)
        if alpha < 1e-8 or a + 1 >= len(q):
            continue
        b = a + 1
        sample = result[a] * (1 - alpha) + result[b] * alpha
        sample[3:7] = root[k].as_quat()[[3, 0, 1, 2]]
        initial_error = np.rad2deg(np.linalg.norm(error(sample, goals[k])))
        if initial_error <= limit_deg:
            continue
        original = result[[a, b]].copy()
        initial = original[:, qi].ravel()
        def residual(x):
            knots = original.copy()
            knots[:, qi] = x.reshape(2, len(qi))
            mixed = sample.copy()
            mixed[qi] = knots[0, qi] * (1 - alpha) + knots[1, qi] * alpha
            errors = [error(knots[0], targets[a]), error(knots[1], targets[b]), error(mixed, goals[k])]
            excess = [e * max(1 - np.deg2rad(limit_deg * .95) / max(np.linalg.norm(e), 1e-12), 0) for e in errors]
            return np.r_[errors[0], errors[1], errors[2] * 2, np.concatenate(excess) * 100, (x - initial) * .001]
        lo = np.tile(knot_lo, 2)
        hi = np.tile(knot_hi, 2)
        opt = least_squares(residual, np.clip(initial, lo + 1e-10, hi - 1e-10), bounds=(lo, hi), max_nfev=200,
                            ftol=1e-11, xtol=1e-11, gtol=1e-11)
        result[np.ix_([a, b], qi)] = opt.x.reshape(2, len(qi))
        repaired.append(dict(output_frame=k, native_frames=[a, b], initial_error_deg=float(initial_error), solver_success=bool(opt.success)))
    return result, repaired


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('input', 'trace', 'model', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--output-fps', type=float, default=50.)
    p.add_argument('--limit-deg', type=float, default=.18)
    p.add_argument('--joint-limit-margin-deg', type=float, default=2.)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / 'csv').mkdir()
    with a.input.open() as f:
        reader = csv.reader(f); header = next(reader); raw = np.array(list(reader), float)
    m = mujoco.MjModel.from_xml_path(str(a.model))
    clean = [n.removesuffix('_dof') for n in header]
    cols = [clean.index(m.joint(j).name) for j in range(1, m.njnt)]
    q = np.c_[raw[:, 1:4], R.from_euler('xyz', raw[:, 4:7]).as_quat()[:, [3, 0, 1, 2]], raw[:, cols]]
    trace = np.load(a.trace)
    if len(q) != len(trace['targets']):
        raise ValueError('Frame mismatch')
    records = {}
    for side in ('Left', 'Right'):
        targets = R.from_quat(trace['targets'][:, trace['names'].tolist().index(side + 'Hand'), 3:])
        q, records[side] = refine(a.model, q, targets, side, float(trace['fps']), a.output_fps, a.limit_deg, a.joint_limit_margin_deg)
    output = raw.copy(); output[:, cols] = q[:, 7:]
    keep = [i for i, name in enumerate(clean) if '_wrist_' not in name]
    assert np.array_equal(output[:, keep], raw[:, keep])
    with (a.out / 'csv' / a.input.name).open('w') as f:
        writer = csv.writer(f); writer.writerow(header); writer.writerows(output)
    (a.out / 'report.json').write_text(json.dumps(dict(source=str(a.input), source_sha256=hashlib.sha256(a.input.read_bytes()).hexdigest(),
        native_fps=float(trace['fps']), output_fps=a.output_fps, limit_deg=a.limit_deg, joint_limit_margin_deg=a.joint_limit_margin_deg, repairs=records,
        non_wrist_columns_bit_preserved=True, requires_independent_full_audit=True), indent=2) + '\n')
    (a.out / 'runner.py').write_bytes(Path(__file__).read_bytes())


if __name__ == '__main__':
    main()
