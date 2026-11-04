"""Resumeable reconstruction/refit of remaining failures. No training admission.

Dry runs only read the inventory. Each worker uses a disposable /tmp directory;
only validated CSV/reference/audit triples survive. Failed cases are recorded in
one SQLite database, without keeping trial trajectories or Markdown reports.
"""
from pathlib import Path
import argparse
import collections
import concurrent.futures
import contextlib
import csv
import fcntl
import hashlib
import importlib.util
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import time
import traceback

HERE = Path(__file__).resolve().parent
BASE = Path('/home/arrc/lib/groot_wbc')
SOURCE = BASE / 'artifacts/h2_bvh_gpu_refine131k_116014_v6'
BUNDLE = BASE / 'artifacts/h2_bvh_gpu_staged_v5'
SCRIPTS = BASE / 'workspace/GR00T-WholeBodyControl-h2-clean/gear_sonic/scripts'
CATALOG = BASE / 'artifacts/h2_accepted52219_compound_v9/catalog.json'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.partial')
    tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def inventory(keys=None):
    retry = read(SOURCE / 'retry_plan.json')['motions']
    catalog = read(CATALOG)
    accepted = {r['key'] for r in catalog}
    resolved = read(HERE / 'unresolved_progress.json')['motions']
    protected = {k for k, r in resolved.items() if r['status'] in (
        'legacy_accepted', 'new_H2_reference_validated',
        'new_BVH_H2_foot_reference_validated')}
    frames = read(BUNDLE / 'length_inventory.json')['frames']
    requested = set(keys) if keys else None
    known = {r['key'] for r in retry}
    if requested and requested - known:
        raise ValueError('Keys outside original failure inventory: ' + repr(sorted(requested - known)))
    rows = []
    counts = collections.Counter()
    for row in retry:
        key = row['key']
        if requested is not None and key not in requested:
            continue
        if key in accepted or key in protected:
            counts['already_accepted_or_resolved'] += 1
            continue
        flags = {reason.split(':')[0] for reason in row['reasons']}
        flags = {'velocity' if x == 'joint_velocity' else x for x in flags}
        rows.append(dict(key=key, frames=frames[key], reasons=row['reasons'],
                         category='+'.join(sorted(flags))))
    rows.sort(key=lambda r: (r['frames'], r['key']))
    return rows, dict(original_failures=len(retry), selected=len(rows),
                      excluded=counts['already_accepted_or_resolved'],
                      categories=dict(collections.Counter(r['category'] for r in rows)),
                      native_frames=sum(r['frames'] for r in rows),
                      scope='All remaining original failures; accepted catalog and resolved eight excluded.')


def worker_init():
    global U, F, E, AUDITOR, API, JOBS
    for var in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[var] = '1'
    common = module('batch_common', SCRIPTS / 'run_h2_foot_preserving_reopt.py')
    API = common.api_for(BUNDLE)
    API.verify_bundle()
    JOBS = {r['key']: r for r in API.jobs_for(SOURCE)}
    U = module('batch_upper', HERE / 'upper_feasibility.py')
    F = module('batch_feet', HERE / 'reconstruct_bvh_feet_v14.py')
    E = module('batch_foot_evaluator', HERE / 'audit_bvh_reconstruction.py')
    AUDITOR = module('batch_independent_auditor', SCRIPTS / 'h2_contact_checked_foot_audit.py')


def write_csv(path, p, q):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = p.raw.copy()
    raw[:, 1:4] = q[:, :3]
    raw[:, p.cols] = q[:, 7:]
    # Keep frame IDs and original Euler root columns exactly as loaded.
    with path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(p.header)
        w.writerows(raw)


def audit(root, key, p, q):
    path = root / 'refined/csv' / f'{key}.csv'
    write_csv(path, p, q)
    AUDITOR.audit_task((str(BUNDLE), str(SOURCE), str(root),
                       dict(JOBS[key], recipe='bvh_h2_batch_reconstruction')))
    report = read(root / 'reports' / f'{key}.json')
    assert report['refined_sha256'] == sha(path)
    return path, report


def physical_projection(p, q):
    """Minimal joint-space projection; finite differences at native 120Hz."""
    import numpy as np
    import osqp
    from scipy import sparse
    n, dim = q[:, 7:].shape
    margin = np.array([np.deg2rad(2) + 1e-6 if any(s in name for s in ('shoulder', 'wrist'))
                       else 1e-7 for name in p.names])
    lo = p.m.jnt_range[1:, 0] + margin
    hi = p.m.jnt_range[1:, 1] - margin
    cap = p.speeds / 120 * .9995
    values = q[:, 7:]
    if (np.all(values >= lo) and np.all(values <= hi)
            and np.all(abs(np.diff(values, axis=0)) <= cap)):
        return q.copy(), dict(changed=False)
    difference = sparse.kron(sparse.diags([-np.ones(n-1), np.ones(n-1)],
                                          [0, 1], shape=(n-1, n)),
                             sparse.eye(dim), format='csc')
    eye = sparse.eye(n * dim, format='csc')
    A = sparse.vstack([eye, difference], format='csc')
    solver = osqp.OSQP()
    solver.setup(P=eye, q=-values.ravel(), A=A,
                 l=np.r_[np.tile(lo, n), -np.tile(cap, n-1)],
                 u=np.r_[np.tile(hi, n), np.tile(cap, n-1)],
                 eps_abs=1e-9, eps_rel=1e-9, max_iter=30000,
                 polishing=True, verbose=False)
    answer = solver.solve()
    if answer.x is None or not np.isfinite(answer.x).all():
        raise ValueError('Joint projection produced no finite candidate')
    result = q.copy()
    result[:, 7:] = answer.x.reshape(n, dim)
    angles = result[:, 7:]
    excess = max(float((lo-angles).max()), float((angles-hi).max()), 0.)
    ratio = float((abs(np.diff(angles, axis=0)) * 120 / p.speeds).max())
    if excess > 1e-8 or ratio > 1.000001:
        raise ValueError(f'Joint projection failed actual gates: excess={excess}, speed={ratio}')
    return result, dict(changed=True, status=answer.info.status,
                        max_joint_move_rad=float(abs(result[:, 7:] - values).max()),
                        speed_ratio=ratio, range_excess_rad=excess)


def foot_rates(path, trace):
    evaluator = E.Evaluation('h2', path, trace)
    return {str(rate): evaluator.evaluate(rate) for rate in (120, 50, 600)}


def feet_pass(rates):
    return all(v['penetrating_samples'] == 0 and v['grounded_target_above_0p5mm'] == 0
               for v in rates.values())


def foot_project(key, source, warm, trace, iterations):
    import numpy as np
    p = F.SourceFootProblem(warm, key=key, source_path=source, trace_path=trace)
    x = p.x0.copy()
    q = p.assemble(x)
    tt = np.arange((p.n-1)*5+1)/600
    _, _, heights = p.fm.poses(F.sample(q, tt))
    _, _, lift = p.goals(tt)
    active = (heights.min(1) < p.floor+p.margin) | np.any(
        (lift > 1e-8) & (heights > p.floor+.00035), axis=1)
    p.add_ground_cuts(tt[active].tolist())
    current = p.measure(x)
    if not current['feasible']:
        raise ValueError('No feasible lower-body initialization: ' + repr(current))
    before = dict(current)
    history = []
    for iteration in range(iterations):
        delta, qp = p.qp(x, p.linearize(x))
        winner = None
        trials = []
        if delta is not None:
            for alpha in (1., .5, .25, .125, .0625, .03125, .015625):
                trial = np.clip(x+alpha*delta, p.low, p.high)
                score = p.measure(trial)
                trials.append(score)
                restoring = current['grounded_target_height_max_mm'] > .45
                improvement = (score['grounded_target_height_max_mm'] < current['grounded_target_height_max_mm']-1e-4
                               if restoring else score['cost'] < current['cost']-1e-5)
                preserved = restoring or score['grounded_target_height_max_mm'] <= max(
                    .45, current['grounded_target_height_max_mm'])+1e-6
                if score['feasible'] and improvement and preserved:
                    winner = trial, score
                    break
        if winner is not None:
            x, current = winner
        q = p.assemble(x)
        _, _, heights = p.fm.poses(F.sample(q, tt))
        floating = np.where(lift > 1e-8, heights-(p.floor+.0004), -np.inf).max(1)
        worst = np.argsort(floating)[::-1]
        added = p.add_ground_cuts([float(tt[k]) for k in worst[:64] if floating[k] > 0])
        if winner is None:
            added += p.add_ground_cuts([t for row in trials for t in row['dense_violations']])
        history.append(dict(iteration=iteration+1, accepted=winner is not None, qp=qp,
                            cost=current['cost'], clearance_mm=current['min_clearance_mm'],
                            grounded_height_mm=current['grounded_target_height_max_mm']))
        if added:
            current = p.measure(x)
        if winner is None and not added:
            break
    # No-contact clips use -inf internally; keep receipts valid JSON.
    for score in (before, current):
        if not np.isfinite(score['grounded_target_height_max_mm']):
            score['grounded_target_height_max_mm'] = None
    return p.assemble(x), dict(before=before, after=current, history=history)


def run_motion(row, config):
    import numpy as np
    key = row['key']
    start = time.time()
    result = dict(key=key, category=row['category'], status='needs_review')
    with tempfile.TemporaryDirectory(prefix='h2_batch_', dir=config['scratch']) as directory:
        root = Path(directory)
        # Keep optimization traces disposable; no interleaved multi-worker stdout.
        with (root/'trial.log').open('w') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                job = JOBS[key]
                if not API.raw_valid(Path(job['work']), job):
                    raise ValueError('Raw BVH/G1 provenance changed')
                original = SOURCE/'refined/csv'/f'{key}.csv'
                original_report = read(SOURCE/'reports'/f'{key}.json')
                original_sha = sha(original)
                assert original_sha == original_report['refined_sha256']
                trace = Path(job['work'])/'h2'/f'{key}_trace.npz'
                pins = {str(p): sha(p) for p in (original, trace)}
                U.ROOT = root
                seed_root = root/'existing_eight_reaudit'
                seed = seed_root/'refined/csv'/f'{key}.csv'
                seed.parent.mkdir(parents=True)
                shutil.copyfile(original, seed)
                AUDITOR.audit_task((str(BUNDLE), str(SOURCE), str(seed_root), job))
                canonical = read(seed_root/'reports'/f'{key}.json')
                p = U.Problem(key, job, arms_only=True)
                tests, _ = p.verify()
                original_q = p.q0.copy()
                original_feet = foot_rates(original, trace)
                q, physical = physical_projection(p, original_q)
                stages = dict(joint_projection=physical)
                warm, warm_report = audit(root/'physical', key, p, q)
                current_feet = foot_rates(warm, trace)
                lower_needed = 'feet' in row['category'] or not feet_pass(current_feet)
                if lower_needed:
                    q, stages['feet'] = foot_project(key, original, warm, trace, config['foot_iterations'])
                    warm, warm_report = audit(root/'lower', key, p, q)
                    current_feet = foot_rates(warm, trace)
                    if not feet_pass(current_feet):
                        result.update(reason='BVH/H2 grounded foot gates failed', feet=current_feet,
                                      seconds=round(time.time()-start, 2))
                        return result
                # Rebuild a feasible seed in the disposable root. The published
                # eight-case directory and source reports are never modified.
                audit(seed_root, key, p, q)
                if 'torso' in row['category']:
                    body = U.Problem(key, job)
                    x, stages['body'] = body.reconstruct(config['iterations'])
                    q = body.assemble(x)
                    audit(seed_root, key, p, q)
                p = U.Problem(key, job, arms_only=True)
                refit_seed_sha = sha(p.seed)
                x, generation = p.anatomical_project(config['iterations'], config['wrist_budget'])
                goal_q = p.assemble(x)
                goal_path, goal_report = audit(root/'goal', key, p, goal_q)
                goal_quality = U.raw_quality_review(goal_report, canonical)
                if not goal_quality['primary_nonregression_pass'] or not goal_report['physical_and_margin_pass']:
                    result.update(reason='Reconstructed goal failed original-BVH primary/physical gates',
                                  primary_regressions=goal_quality['primary_regressions'],
                                  physical_pass=goal_report['physical_and_margin_pass'],
                                  seconds=round(time.time()-start, 2))
                    return result
                # Freeze the target before a distinct arms-only refit initialized
                # from the feasible seed, not from the completed goal trajectory.
                reference = root/f'{key}.npz'
                metadata = dict(contract='frozen_projected_h2_upper_v1', key=key, fps=120,
                                raw_trace_sha256=sha(trace), model_sha256=sha(BUNDLE/'code/gear_sonic/data/assets/robot_description/mjcf/h2.xml'),
                                source_csv_sha256=sha(goal_path), baseline_sha256=original_sha,
                                wrist_correction_budget_deg=config['wrist_budget'])
                np.savez_compressed(reference, qpos=goal_q, metadata=np.array(json.dumps(metadata)))
                U.attach_reference(p, reference)
                x, refit = p.reconstruct(config['refit_iterations'])
                candidate_q = p.assemble(x)
                candidate, report = audit(root/'final', key, p, candidate_q)
                fidelity = U.fixed_reference_check(p, candidate, reference)
                quality = U.raw_quality_review(report, canonical)
                final_feet = foot_rates(candidate, trace)
                lower_columns = [7+i for i, name in enumerate(p.names)
                                 if any(s in name for s in ('hip', 'knee', 'ankle'))]
                unchanged_foot_after_lower = np.array_equal(candidate_q[:, :7], p.q0[:, :7]) and np.array_equal(
                    candidate_q[:, lower_columns], p.q0[:, lower_columns])
                root_rotation_exact = np.array_equal(candidate_q[:, 3:7], original_q[:, 3:7])
                foot_improvement = lower_needed and (
                    not feet_pass(original_feet) or
                    stages['feet']['after']['cost'] < stages['feet']['before']['cost']-1e-5)
                changed = not np.array_equal(original_q, candidate_q)
                new_pass = bool(fidelity['passed'] and quality['primary_nonregression_pass']
                                and (quality['primary_improvement_pass'] or foot_improvement)
                                and report['physical_and_margin_pass'] and feet_pass(final_feet)
                                and unchanged_foot_after_lower and root_rotation_exact and changed)
                legacy_pass = report['status'] == 'accepted' and feet_pass(final_feet)
                result.update(status='legacy_accepted' if legacy_pass else 'new_H2_reference_validated' if new_pass else 'needs_review',
                              new_reference_pass=new_pass, legacy_full_pass=legacy_pass,
                              legacy_failed_checks=sum(not c['passed'] for c in report['checks']),
                              physical_pass=report['physical_and_margin_pass'],
                              fixed_reference=fidelity, raw_quality_review=quality,
                              feet=final_feet, foot_reconstruction_applied=lower_needed,
                              native_frames=row['frames'])
                assert sha(p.seed) == refit_seed_sha
                assert all(sha(path) == value for path, value in pins.items())
                if new_pass or legacy_pass:
                    out = Path(config['out'])
                    dest = out/'validated/csv'/f'{key}.csv'
                    ref_dest = out/'validated/references'/f'{key}.npz'
                    rp = out/'validated/reports'/f'{key}.json'
                    report['batch_validation'] = dict(result,
                        source_pins=pins, generation=generation, refit=refit,
                        stages=stages, tests=tests, refit_seed_sha256=refit_seed_sha,
                        no_training_admission=True,
                        validation_scope='Kinematic BVH/H2 reference only; dynamics and comprehensive self collision are untested.')
                    report['batch_validation']['fixed_reference']['reference'] = str(ref_dest)
                    for src, dst in ((candidate, dest), (reference, ref_dest)):
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        if dst.exists() and sha(dst) != sha(src):
                            raise ValueError('Existing validated output differs; refusing overwrite: '+str(dst))
                        tmp = dst.with_suffix(dst.suffix+'.partial')
                        shutil.copyfile(src, tmp)
                        tmp.replace(dst)
                    report['validated_source_output'] = report['output']
                    report['output'] = str(dest)
                    save(rp, report)
                    result.update(csv=str(dest), csv_sha256=sha(dest), reference=str(ref_dest),
                                  reference_sha256=sha(ref_dest), report=str(rp), report_sha256=sha(rp))
            except Exception as error:
                result.update(status='error', reason=f'{type(error).__name__}: {error}',
                              traceback=traceback.format_exc(limit=6))
                if (root/'trial.log').exists():
                    log.flush()
                    result['diagnostic_tail'] = (root/'trial.log').read_text()[-2500:]
    # Avoid duplicating full per-stat/per-rate reports in the shared database.
    if 'raw_quality_review' in result:
        review = result['raw_quality_review']
        result['raw_quality_review'] = {k: v for k, v in review.items() if k not in ('primary_checks', 'raw_wrist_checks')}
    result['seconds'] = round(time.time()-start, 2)
    return result


def print_status(db):
    counts = dict(db.execute('SELECT state, count(*) FROM motions GROUP BY state'))
    last = db.execute('SELECT key,state FROM motions WHERE state != ? ORDER BY updated DESC LIMIT 1', ('pending',)).fetchone()
    print(json.dumps(dict(counts=counts, last=last), ensure_ascii=False), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--all-failed', action='store_true')
    ap.add_argument('--keys', nargs='+')
    ap.add_argument('--out', type=Path)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--status', action='store_true')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--iterations', type=int, default=30)
    ap.add_argument('--refit-iterations', type=int, default=32)
    ap.add_argument('--foot-iterations', type=int, default=24)
    ap.add_argument('--wrist-budget-deg', type=float, default=15.)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--scratch', type=Path, default=Path('/tmp'))
    a = ap.parse_args()
    if a.status:
        if not a.out or not (a.out/'state.sqlite3').exists():
            ap.error('--status requires an existing --out/state.sqlite3')
        with sqlite3.connect(f'file:{a.out}/state.sqlite3?mode=ro', uri=True) as db:
            print_status(db)
        return
    if not (a.all_failed or a.keys) or (a.all_failed and a.keys):
        ap.error('Choose --all-failed or --keys KEY ...')
    if min(a.workers, a.iterations, a.refit_iterations, a.foot_iterations) < 1 or a.limit < 0 or not 0 < a.wrist_budget_deg <= 15:
        ap.error('Positive sizes required; hand reconstruction budget must be <=15deg')
    rows, summary = inventory(a.keys)
    if a.limit:
        rows = rows[:a.limit]
        summary['execution_limit'] = a.limit
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if a.dry_run:
        return
    if a.out is None:
        ap.error('Execution requires --out')
    out = a.out.resolve()
    if out == HERE.resolve() or HERE.resolve() in out.parents:
        ap.error('Use a separate output directory outside the completed eight-case experiment')
    out.mkdir(parents=True, exist_ok=True)
    lock = (out/'run.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        ap.error('Another batch is already running in this output directory')
    pins = {str(p): sha(p) for p in (CATALOG,
        BASE/'artifacts/h2_accepted52217_recovery_v6/catalog.json',
        SOURCE/'complete.json', SOURCE/'retry_plan.json', HERE/'unresolved_progress.json')}
    code = {str(p): sha(p) for p in (Path(__file__), HERE/'upper_feasibility.py',
        HERE/'native_projection.py', HERE/'reconstruct_bvh_feet_v14.py',
        HERE/'audit_bvh_reconstruction.py', SCRIPTS/'h2_contact_checked_foot_audit.py',
        SCRIPTS/'h2_expansion_audit.py', SCRIPTS/'h2_experimental_foot_audit.py',
        BUNDLE/'code/gear_sonic/scripts/audit_g1_style_retarget.py')}
    config = dict(out=str(out), scratch=str(a.scratch), iterations=a.iterations,
                  refit_iterations=a.refit_iterations, foot_iterations=a.foot_iterations,
                  wrist_budget=a.wrist_budget_deg)
    manifest = dict(schema='h2_bvh_h2_failure_batch_v1', pins=pins, code=code,
                    config=config, keys_sha256=hashlib.sha256('\n'.join(r['key'] for r in rows).encode()).hexdigest(),
                    summary=summary, training_admission_performed=False, markdown_reports=False)
    if (out/'manifest.json').exists():
        if not a.resume:
            ap.error('Existing output requires --resume')
        if read(out/'manifest.json') != manifest:
            ap.error('Selection, source or settings changed; use a different --out')
    else:
        save(out/'manifest.json', manifest)
    db = sqlite3.connect(out/'state.sqlite3')
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS motions (key TEXT PRIMARY KEY, category TEXT, frames INTEGER, state TEXT, result TEXT, updated REAL)')
    db.executemany('INSERT OR IGNORE INTO motions VALUES (?,?,?,?,?,?)',
                   [(r['key'], r['category'], r['frames'], 'pending', None, 0.) for r in rows])
    db.execute("UPDATE motions SET state='pending' WHERE state='running'")
    db.commit()
    pending = {r[0] for r in db.execute("SELECT key FROM motions WHERE state='pending'")}
    # Verify all saved successes on resume; a stale flag cannot skip missing data.
    for key, payload in db.execute("SELECT key,result FROM motions WHERE state IN ('legacy_accepted','new_H2_reference_validated')"):
        record = json.loads(payload)
        for name in ('csv', 'reference', 'report'):
            if sha(record[name]) != record[name+'_sha256']:
                raise ValueError('Resumed output changed: '+key+' '+name)
    for var in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[var] = '1'
    from multiprocessing import get_context
    iterator = iter(r for r in rows if r['key'] in pending)
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,
            mp_context=get_context('spawn'), initializer=worker_init) as pool:
        active = {}
        def submit():
            row = next(iterator, None)
            if row is None:
                return False
            db.execute("UPDATE motions SET state='running',updated=? WHERE key=?", (time.time(), row['key']))
            db.commit()
            active[pool.submit(run_motion, row, config)] = row
            return True
        for _ in range(a.workers):
            submit()
        heartbeat = time.time()
        while active:
            completed, _ = concurrent.futures.wait(active, timeout=30,
                return_when=concurrent.futures.FIRST_COMPLETED)
            for future in completed:
                row = active.pop(future)
                try:
                    result = future.result()
                except Exception as error:
                    result = dict(key=row['key'], status='error', reason=repr(error))
                db.execute('UPDATE motions SET state=?,result=?,updated=? WHERE key=?',
                    (result['status'], json.dumps(result, allow_nan=False), time.time(), row['key']))
                db.commit()
                print(json.dumps({k: result[k] for k in ('key', 'status', 'seconds', 'reason') if k in result}), flush=True)
                submit()
            if completed or time.time()-heartbeat >= 30:
                print_status(db)
                heartbeat = time.time()
    assert all(sha(path) == value for path, value in pins.items()), 'Protected inputs changed'
    print_status(db)
    counts = dict(db.execute('SELECT state,count(*) FROM motions GROUP BY state'))
    save(out/'complete.json', dict(counts=counts, training_admission_performed=False,
                                  failed_cases_keep_only_database_diagnostics=True))
    db.close()


if __name__ == '__main__':
    main()
