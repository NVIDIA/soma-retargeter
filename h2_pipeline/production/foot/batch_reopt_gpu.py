"""GPU geometry backend for the existing failure batch and state database.

Completed CPU outputs are verified and retained. CUDA float64 FK/Jacobians feed
the unchanged CPU constrained QPs; final admission remains CPU MuJoCo only.
"""
from pathlib import Path
import argparse
import concurrent.futures
import fcntl
import json
import multiprocessing
import os
import signal
import sqlite3
import time
import batch_reopt as B

G = None


def initialize(devices):
    global G
    import torch
    import gpu_acceleration
    identity = multiprocessing.current_process()._identity[-1]
    device = 'cuda:'+str(devices[(identity-1) % len(devices)])
    torch.cuda.set_device(device)
    B.worker_init()
    G = gpu_acceleration.install(B, device)


def run(row, config):
    import torch
    before_calls, before_seconds = G.calls, G.seconds
    result = B.run_motion(row, config)
    backend = dict(name='cuda_float64_geometry_cpu_constrained_qp', device=G.device,
                   gpu=torch.cuda.get_device_name(G.device),
                   upper_gpu_evaluations=G.calls-before_calls,
                   upper_gpu_evaluation_seconds=G.seconds-before_seconds,
                   final_admission='Unchanged independent CPU MuJoCo')
    result['execution_backend'] = backend
    if result.get('report'):
        report = B.read(result['report'])
        report['batch_validation']['execution_backend'] = backend
        B.save(Path(result['report']), report)
        result['report_sha256'] = B.sha(result['report'])
    return result


def recover(db, out):
    """Recover only uncommitted in-flight receipts after stopping CPU workers."""
    recovered = 0
    for key, in list(db.execute("SELECT key FROM motions WHERE state='running'")):
        rp = out/'validated/reports'/f'{key}.json'
        csv = out/'validated/csv'/f'{key}.csv'
        ref = out/'validated/references'/f'{key}.npz'
        if all(p.exists() for p in (rp, csv, ref)):
            report = B.read(rp)
            result = report['batch_validation']
            if (result['status'] in ('legacy_accepted', 'new_H2_reference_validated')
                    and B.sha(csv) == report['refined_sha256']
                    and B.sha(ref) == result['fixed_reference']['reference_sha256']):
                result.update(csv=str(csv), csv_sha256=B.sha(csv), reference=str(ref),
                              reference_sha256=B.sha(ref), report=str(rp), report_sha256=B.sha(rp),
                              recovered_from_completed_worker=True)
                db.execute('UPDATE motions SET state=?,result=?,updated=? WHERE key=?',
                    (result['status'], json.dumps(result, allow_nan=False), time.time(), key))
                recovered += 1
                continue
            raise ValueError('In-flight output receipt failed: '+key)
        db.execute("UPDATE motions SET state='pending',result=NULL WHERE key=?", (key,))
    db.commit()
    return recovered


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--all-failed', action='store_true')
    ap.add_argument('--keys', nargs='+')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--devices', default='0,1')
    ap.add_argument('--iterations', type=int, default=30)
    ap.add_argument('--refit-iterations', type=int, default=32)
    ap.add_argument('--foot-iterations', type=int, default=24)
    ap.add_argument('--wrist-budget-deg', type=float, default=15.)
    ap.add_argument('--scratch', type=Path, default=Path('/tmp'))
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    if bool(a.all_failed) == bool(a.keys):
        ap.error('Choose --all-failed or --keys')
    if min(a.workers, a.iterations, a.refit_iterations, a.foot_iterations) < 1 or not 0 < a.wrist_budget_deg <= 15:
        ap.error('Positive sizes and hand budget <=15deg required')
    devices = [int(s) for s in a.devices.split(',')]
    if not devices or len(set(devices)) != len(devices):
        ap.error('Specify distinct CUDA device indices')
    rows, summary = B.inventory(a.keys)
    print(json.dumps(summary), flush=True)
    if a.dry_run:
        return
    out = a.out.resolve()
    if out == B.HERE.resolve() or B.HERE.resolve() in out.parents:
        ap.error('Use a separate batch output directory')
    out.mkdir(parents=True, exist_ok=True)
    lock = (out/'run.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        ap.error('A CPU/GPU controller still owns this output directory')
    config = dict(out=str(out), scratch=str(a.scratch), iterations=a.iterations,
                  refit_iterations=a.refit_iterations, foot_iterations=a.foot_iterations,
                  wrist_budget=a.wrist_budget_deg)
    pins = {str(p): B.sha(p) for p in (B.CATALOG,
        B.BASE/'artifacts/h2_accepted52217_recovery_v6/catalog.json',
        B.SOURCE/'complete.json', B.SOURCE/'retry_plan.json', B.HERE/'unresolved_progress.json')}
    code_paths = [Path(__file__), B.HERE/'gpu_acceleration.py', B.HERE/'batch_reopt.py',
        B.HERE/'upper_feasibility.py', B.HERE/'native_projection.py',
        B.HERE/'reconstruct_bvh_feet_v14.py', B.HERE/'audit_bvh_reconstruction.py',
        B.SCRIPTS/'h2_contact_checked_foot_audit.py', B.SCRIPTS/'h2_expansion_audit.py',
        B.SCRIPTS/'h2_experimental_foot_audit.py',
        B.BUNDLE/'code/gear_sonic/scripts/h2_gpu_reference_ik.py',
        B.BUNDLE/'code/gear_sonic/scripts/h2_waist_gpu_ik.py']
    manifest = dict(schema='h2_failure_cuda_geometry_v1', config=config, pins=pins,
                    code={str(p): B.sha(p) for p in code_paths}, summary=summary,
                    training_admission_performed=False,
                    cpu_manifest_sha256=B.sha(out/'manifest.json') if (out/'manifest.json').exists() else None)
    mp = out/'gpu_manifest.json'
    if mp.exists():
        if not a.resume or B.read(mp) != manifest:
            ap.error('Existing GPU run requires --resume with unchanged code and criteria')
    else:
        if (out/'state.sqlite3').exists():
            if not a.resume:
                ap.error('Existing CPU state requires --resume')
            original = B.read(out/'manifest.json')
            assert original['config'] == config, 'CPU/GPU reconstruction settings differ'
            assert original['pins'] == pins, 'Protected CPU source pins changed'
        B.save(mp, manifest)
    db = sqlite3.connect(out/'state.sqlite3')
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS motions (key TEXT PRIMARY KEY, category TEXT, frames INTEGER, state TEXT, result TEXT, updated REAL)')
    db.executemany('INSERT OR IGNORE INTO motions VALUES (?,?,?,?,?,?)',
        [(r['key'], r['category'], r['frames'], 'pending', None, 0.) for r in rows])
    db.commit()
    assert {r[0] for r in db.execute('SELECT key FROM motions')} == {r['key'] for r in rows}, 'Selection changed'
    count = recover(db, out)
    for key, payload in db.execute("SELECT key,result FROM motions WHERE state IN ('legacy_accepted','new_H2_reference_validated')"):
        record = json.loads(payload)
        for field in ('csv', 'reference', 'report'):
            if B.sha(record[field]) != record[field+'_sha256']:
                raise ValueError('Previously validated file changed: '+key+' '+field)
    print(json.dumps(dict(stage='gpu_resume', recovered_uncommitted_successes=count,
                          devices=devices, workers=a.workers)), flush=True)
    B.print_status(db)
    pending = {r[0] for r in db.execute("SELECT key FROM motions WHERE state='pending'")}
    iterator = iter(r for r in rows if r['key'] in pending)
    for var in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[var] = '1'
    stopping = False
    def request_stop(signum, frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    drain = out/'drain.request'
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,
            mp_context=multiprocessing.get_context('spawn'), initializer=initialize,
            initargs=(devices,)) as pool:
        active = {}
        def submit():
            if stopping or drain.exists():
                return
            row = next(iterator, None)
            if row is None:
                return
            db.execute("UPDATE motions SET state='running',updated=? WHERE key=?", (time.time(), row['key']))
            db.commit()
            active[pool.submit(run, row, config)] = row
        for _ in range(a.workers):
            submit()
        while active:
            done, _ = concurrent.futures.wait(active, timeout=30,
                return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                row = active.pop(future)
                try:
                    result = future.result()
                except Exception as error:
                    result = dict(key=row['key'], status='error', reason=repr(error))
                db.execute('UPDATE motions SET state=?,result=?,updated=? WHERE key=?',
                    (result['status'], json.dumps(result, allow_nan=False), time.time(), row['key']))
                db.commit()
                print(json.dumps({k: result[k] for k in ('key', 'status', 'seconds', 'reason', 'execution_backend') if k in result}), flush=True)
                submit()
            B.print_status(db)
    assert all(B.sha(p) == value for p, value in pins.items()), 'Protected inputs changed'
    counts = dict(db.execute('SELECT state,count(*) FROM motions GROUP BY state'))
    B.save(out/'gpu_progress.json', dict(counts=counts, drained=stopping or drain.exists(),
                                        training_admission_performed=False))
    if counts.get('pending', 0) == 0 and counts.get('running', 0) == 0:
        B.save(out/'complete.json', dict(counts=counts, backend='cuda_geometry_cpu_constrained_qp',
                                       training_admission_performed=False))
    B.print_status(db)
    db.close()


if __name__ == '__main__':
    main()
