"""Original SQP: shared CUDA geometry and bounded parallel CPU direct QPs."""
from pathlib import Path
import argparse
import ctypes
import gc
import json
import multiprocessing
import os
import pickle
import secrets
import subprocess
import sys
import tempfile
import threading
import time
import batch_reopt as B
import batch_reopt_gpu as D
import shared_gpu_geometry as S

QP = None


class DirectQPStats:
    def __init__(self, original, semaphore):
        self.original, self.semaphore = original, semaphore
        self.calls = 0
        self.seconds = 0.


class DirectQP:
    def __init__(self):
        self.solver = QP.original(algebra='builtin')
        self.held = False

    def setup(self, *args, **kwargs):
        self.begin = time.perf_counter()
        QP.semaphore.acquire()
        self.held = True
        try:
            self.solver.setup(*args, **kwargs)
        except BaseException:
            self.release()
            raise

    def release(self):
        if self.held:
            self.held = False
            QP.semaphore.release()

    def solve(self, **kwargs):
        try:
            return self.solver.solve(**kwargs)
        finally:
            QP.calls += 1
            QP.seconds += time.perf_counter()-self.begin
            self.release()

    def __del__(self):
        self.release()


def initialize(devices):
    global QP
    import osqp
    S.install_compiled_model()
    B.worker_init()
    config = B.read(os.environ['H2_HYBRID_CONFIG'])
    device = devices[(multiprocessing.current_process()._identity[-1]-1) % len(devices)]
    entry = next(r for r in config['actors'] if r['device'] == device)
    D.G = S.install(B, entry['endpoint'], config['authkey'], 'cuda:'+str(device))
    with Path(config['semaphore']).open('rb') as file:
        semaphore = pickle.load(file)
    QP = DirectQPStats(osqp.OSQP, semaphore)
    osqp.OSQP = DirectQP
    gc.collect(); ctypes.CDLL('libc.so.6').malloc_trim(0)


def run(row, config):
    calls, seconds = QP.calls, QP.seconds
    geometry_calls, geometry_seconds = D.G.calls, D.G.seconds
    try:
        result = B.run_motion(row, config)
        backend = dict(name='shared_cuda_geometry_parallel_cpu_direct_qp', device=D.G.device,
                       original_objective_and_hard_constraints=True,
                       qp_calls=QP.calls-calls, qp_wait_and_solve_seconds=QP.seconds-seconds,
                       upper_gpu_evaluations=D.G.calls-geometry_calls,
                       upper_gpu_evaluation_seconds=D.G.seconds-geometry_seconds,
                       final_admission='Unchanged independent CPU MuJoCo')
        result['execution_backend'] = backend
        if result.get('report'):
            report = B.read(result['report'])
            report['batch_validation']['execution_backend'] = backend
            B.save(Path(result['report']), report)
            result['report_sha256'] = B.sha(result['report'])
        return result
    finally:
        gc.collect(); ctypes.CDLL('libc.so.6').malloc_trim(0)


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('--qp-slots', type=int, default=48)
    args, rest = ap.parse_known_args()
    if args.qp_slots < 1:
        raise ValueError('Positive QP concurrency required')
    if '--dry-run' in rest or '--help' in rest:
        sys.argv = [sys.argv[0], *rest]; D.main(); return
    out = Path(rest[rest.index('--out')+1]).resolve()
    devices = [int(x) for x in rest[rest.index('--devices')+1].split(',')] if '--devices' in rest else [0, 1]
    out.mkdir(parents=True, exist_ok=True)
    manifest = dict(schema='shared_cuda_geometry_cpu_direct_v1', qp_slots=args.qp_slots,
                    objective_constraints_and_admission='Unchanged original pipeline',
                    code={str(p): B.sha(p) for p in [Path(__file__), B.HERE/'shared_gpu_geometry.py',
                         B.HERE/'gpu_acceleration.py', B.HERE/'batch_reopt.py',
                         B.HERE/'upper_feasibility.py', B.HERE/'reconstruct_bvh_feet_v14.py']})
    receipt = out/'hybrid_execution_manifest.json'
    if receipt.exists() and B.read(receipt) != manifest:
        raise ValueError('Hybrid code/settings differ from recorded execution')
    if not receipt.exists():
        B.save(receipt, manifest)
    with tempfile.TemporaryDirectory(prefix='h2_hybrid_', dir='/tmp') as directory:
        scratch = Path(directory)
        S.prepare_compiled_model(B, scratch)
        with multiprocessing.Manager() as manager:
            semaphore_path = scratch/'qp_slots.pkl'
            semaphore = manager.BoundedSemaphore(args.qp_slots)
            with semaphore_path.open('wb') as file:
                pickle.dump(semaphore, file)
            semaphore_path.chmod(0o600)
            config = dict(authkey=secrets.token_hex(32), semaphore=str(semaphore_path),
                          actors=[dict(device=device, endpoint=str(scratch/f'gpu{device}.sock'),
                                       ready=str(scratch/f'gpu{device}.ready')) for device in devices])
            config_path = scratch/'config.json'; config_path.write_text(json.dumps(config)); config_path.chmod(0o600)
            os.environ['H2_HYBRID_CONFIG'] = str(config_path)
            processes, logs = [], []
            try:
                for entry in config['actors']:
                    log = (out/f"shared_geometry_gpu{entry['device']}.log").open('a'); logs.append(log)
                    processes.append(subprocess.Popen([sys.executable, '-u', str(B.HERE/'shared_gpu_geometry.py'),
                        '--config', str(config_path), '--device', str(entry['device'])], env=dict(os.environ),
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT))
                deadline = time.monotonic()+90
                while not all(Path(r['ready']).exists() for r in config['actors']):
                    if any(p.poll() is not None for p in processes):
                        raise RuntimeError('Shared CUDA geometry actor failed')
                    if time.monotonic() > deadline:
                        raise TimeoutError('Shared geometry startup timed out')
                    time.sleep(.2)
                print(json.dumps(dict(stage='hybrid_actors_ready', devices=devices,
                                      qp_solver='original CPU direct OSQP', qp_slots=args.qp_slots)), flush=True)
                finished = threading.Event()
                def watch():
                    while not finished.wait(2):
                        if any(p.poll() is not None for p in processes):
                            (out/'drain.request').touch()
                            return
                threading.Thread(target=watch, daemon=True).start()
                D.initialize, D.run = initialize, run
                sys.argv = [sys.argv[0], *rest]
                try:
                    D.main()
                finally:
                    finished.set()
            finally:
                for p, entry in zip(processes, config['actors']):
                    if p.poll() is None and Path(entry['endpoint']).exists():
                        try:
                            S.stop(entry['endpoint'], config['authkey'])
                        except (OSError, EOFError):
                            p.terminate()
                    try:
                        p.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        p.kill()
                for log in logs:
                    log.close()


if __name__ == '__main__':
    main()
