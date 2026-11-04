"""Distribute unchanged geometry requests across several processes per GPU."""
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
import batch_reopt_hybrid as H
import shared_gpu_geometry as S


def initialize(devices):
    import osqp
    S.install_compiled_model()
    B.worker_init()
    config = B.read(os.environ['H2_PARALLEL_GEOMETRY_CONFIG'])
    index = (multiprocessing.current_process()._identity[-1]-1) % len(config['actors'])
    entry = config['actors'][index]
    D.G = S.install(B, entry['endpoint'], config['authkey'], 'cuda:'+str(entry['device']))
    with Path(config['semaphore']).open('rb') as file:
        semaphore = pickle.load(file)
    H.QP = H.DirectQPStats(osqp.OSQP, semaphore)
    osqp.OSQP = H.DirectQP
    gc.collect(); ctypes.CDLL('libc.so.6').malloc_trim(0)


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('--qp-slots', type=int, default=48)
    ap.add_argument('--geometry-workers-per-gpu', type=int, default=4)
    args, rest = ap.parse_known_args()
    if min(args.qp_slots, args.geometry_workers_per_gpu) < 1:
        raise ValueError('Positive concurrency required')
    if '--dry-run' in rest or '--help' in rest:
        sys.argv = [sys.argv[0], *rest]; D.main(); return
    out = Path(rest[rest.index('--out')+1]).resolve()
    devices = [int(x) for x in rest[rest.index('--devices')+1].split(',')] if '--devices' in rest else [0, 1]
    out.mkdir(parents=True, exist_ok=True)
    manifest = dict(schema='parallel_original_cuda_geometry_cpu_direct_v1',
                    qp_slots=args.qp_slots, geometry_workers_per_gpu=args.geometry_workers_per_gpu,
                    original_objective_constraints_model_and_admission=True,
                    code={str(p): B.sha(p) for p in [Path(__file__), B.HERE/'batch_reopt_hybrid.py',
                        B.HERE/'shared_gpu_geometry.py', B.HERE/'gpu_acceleration.py',
                        B.HERE/'batch_reopt.py', B.HERE/'upper_feasibility.py',
                        B.HERE/'reconstruct_bvh_feet_v14.py']})
    receipt = out/'parallel_geometry_manifest.json'
    if receipt.exists() and B.read(receipt) != manifest:
        raise ValueError('Parallel geometry settings/code differ from recorded execution')
    if not receipt.exists():
        B.save(receipt, manifest)
    with tempfile.TemporaryDirectory(prefix='h2_parallel_geometry_', dir='/tmp') as directory:
        scratch = Path(directory)
        S.prepare_compiled_model(B, scratch)
        with multiprocessing.Manager() as manager:
            semaphore = manager.BoundedSemaphore(args.qp_slots)
            semaphore_path = scratch/'qp_slots.pkl'
            with semaphore_path.open('wb') as file:
                pickle.dump(semaphore, file)
            semaphore_path.chmod(0o600)
            config = dict(authkey=secrets.token_hex(32), semaphore=str(semaphore_path),
                actors=[dict(device=device, instance=instance,
                    endpoint=str(scratch/f'gpu{device}_{instance}.sock'),
                    ready=str(scratch/f'gpu{device}_{instance}.ready'))
                    for instance in range(args.geometry_workers_per_gpu) for device in devices])
            path = scratch/'config.json'; path.write_text(json.dumps(config)); path.chmod(0o600)
            os.environ['H2_PARALLEL_GEOMETRY_CONFIG'] = str(path)
            processes, logs = [], []
            try:
                for entry in config['actors']:
                    # The original server selects its single configured device;
                    # a private one-entry config identifies each replica.
                    actor_config = scratch/f"gpu{entry['device']}_{entry['instance']}.json"
                    actor_config.write_text(json.dumps(dict(authkey=config['authkey'], actors=[entry])))
                    actor_config.chmod(0o600)
                    log = (out/f"shared_geometry_gpu{entry['device']}.log").open('a'); logs.append(log)
                    processes.append(subprocess.Popen([sys.executable, '-u', str(B.HERE/'shared_gpu_geometry.py'),
                        '--config', str(actor_config), '--device', str(entry['device'])], env=dict(os.environ),
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT))
                deadline = time.monotonic()+120
                while not all(Path(entry['ready']).exists() for entry in config['actors']):
                    if any(p.poll() is not None for p in processes):
                        raise RuntimeError('Parallel geometry server failed to start')
                    if time.monotonic() > deadline:
                        raise TimeoutError('Parallel geometry startup timed out')
                    time.sleep(.2)
                print(json.dumps(dict(stage='parallel_geometry_ready', devices=devices,
                    geometry_workers_per_gpu=args.geometry_workers_per_gpu,
                    qp_slots=args.qp_slots, qp_solver='Unchanged CPU direct OSQP')), flush=True)
                finished = threading.Event()
                def watch():
                    while not finished.wait(2):
                        if any(p.poll() is not None for p in processes):
                            (out/'drain.request').touch()
                            print('Geometry server exited; stopping new work', flush=True)
                            return
                threading.Thread(target=watch, daemon=True).start()
                D.initialize, D.run = initialize, H.run
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
