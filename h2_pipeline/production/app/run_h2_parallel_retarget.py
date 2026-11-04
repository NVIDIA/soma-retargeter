"""Shard an existing H2 v0.2 job across CPU processes sharing ONE GPU.

Default is a read-only plan. --execute is required to create shards/run IK.
Retarget mathematics stays in upstream bvh_to_csv_converter.py. Existing CSVs
are skipped only after schema, finite values, frame indices/count validation.
Incomplete CSVs are reported, never silently overwritten. New CSVs are
published atomically, so interruption cannot leave a truncated final file.
"""
import argparse
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import runpy
import signal
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

REPO = Path(__file__).resolve().parents[1]
ROOT_HEADER = ['Frame', 'root_translateX', 'root_translateY', 'root_translateZ',
               'root_rotateX', 'root_rotateY', 'root_rotateZ']


def bvh_frames(path):
    with path.open() as f:
        for line in f:
            if line.strip().startswith('Frames:'):
                count = int(line.split(':', 1)[1])
                if count < 2:
                    raise ValueError(f'Too few frames: {path}')
                return count
    raise ValueError(f'Missing Frames header: {path}')


def validate_csv(path, frames, joint_names):
    try:
        with path.open(newline='') as f:
            rows = csv.reader(f)
            header = next(rows)
            if (header[:7] != ROOT_HEADER or len(header) != 38
                    or set(header[7:]) != set(joint_names)):
                return False, 'header / joint names'
            count = 0
            for count, row in enumerate(rows, 1):
                if len(row) != 38:
                    return False, f'columns at row {count}'
                values = [float(v) for v in row]
                if not all(math.isfinite(v) for v in values) or values[0] != count - 1:
                    return False, f'nonfinite / frame index at row {count}'
            if count != frames:
                return False, f'frame count {count} != {frames}'
    except (OSError, ValueError, StopIteration) as e:
        return False, str(e)
    return True, 'complete'


def other_writers(export_dir):
    """Refuse to overlap the original single-process converter or this runner."""
    found = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            args = (proc / 'cmdline').read_bytes().decode().split('\0')
            if not any(Path(a).name in ('bvh_to_csv_converter.py', Path(__file__).name)
                       for a in args):
                continue
            if '--config' not in args:
                continue
            config = Path(args[args.index('--config') + 1])
            if not config.is_absolute():
                config = (proc / 'cwd').resolve() / config
            data = json.loads(config.read_text())
            output = Path(data['export_folder'])
            if not output.is_absolute():
                output = (proc / 'cwd').resolve() / output
            if output.resolve() == export_dir:
                found.append(int(proc.name))
        except (OSError, ValueError, KeyError, IndexError):
            continue
    return found


def worker(config):
    # Install only an atomic output adapter, then run the UNCHANGED official CLI.
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    import soma_retargeter.io.csv as csv_io
    original_save = csv_io.save_csv

    def atomic_save(file_path, actuated_joint_names, buffer):
        dst = Path(file_path)
        if dst.exists():
            raise FileExistsError(f'Refusing concurrent overwrite: {dst}')
        fd, tmp = tempfile.mkstemp(prefix='.' + dst.stem + '.', suffix='.partial', dir=dst.parent)
        os.close(fd)
        try:
            original_save(tmp, actuated_joint_names, buffer)
            ok, reason = validate_csv(Path(tmp), buffer.num_frames, actuated_joint_names)
            if not ok:
                raise ValueError(f'Invalid new output {dst}: {reason}')
            # Same-filesystem hard link publishes a complete file without replacing any file.
            os.link(tmp, dst)
        finally:
            Path(tmp).unlink(missing_ok=True)

    csv_io.save_csv = atomic_save
    sys.argv = ['bvh_to_csv_converter.py', '--config', str(config), '--viewer', 'null',
                '--device', 'cuda:0']
    runpy.run_path(str(REPO / 'app/bvh_to_csv_converter.py'), run_name='__main__')


def stop_children(children):
    for child in children:
        if child.poll() is None:
            child.send_signal(signal.SIGINT)
    deadline = time.monotonic() + 10
    for child in children:
        try:
            child.wait(timeout=max(.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16, help='IK microbatch PER worker')
    parser.add_argument('--gpu', default='1', help='Physical GPU index; workers see cuda:0')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.config = args.config.resolve()
    if args.worker:
        worker(args.config)
        return
    if args.workers < 1 or args.batch_size < 1 or not args.gpu.isdecimal():
        parser.error('workers/batch-size must be positive; gpu must be one numeric index')
    cfg = json.loads(args.config.read_text())
    if cfg['retarget_target'] != 'unitree_h2' or cfg.get('extra_robot_paths'):
        raise ValueError('This runner requires official bundled unitree_h2 without custom robot paths')
    source, output = Path(cfg['import_folder']), Path(cfg['export_folder'])
    if not source.is_absolute() or not output.is_absolute():
        raise ValueError('Use absolute import_folder/export_folder paths')
    output = output.resolve()
    writers = other_writers(output)
    if writers and args.execute:
        raise RuntimeError(f'Existing writer PIDs {writers}; stop the old run in its terminal first')
    if writers:
        print(f'WARNING: running writers {writers}; this snapshot may change.', flush=True)
    selection = json.loads((args.config.parent / 'selection.json').read_text())
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    if selection['retarget_commit'] != commit:
        raise ValueError('Retarget commit differs from original job; use a separate output directory')
    all_files = sorted(source.rglob('*.bvh'), key=lambda p: p.relative_to(source).as_posix())
    expected = {m['bvh'] for m in selection['motions']}
    if len(all_files) != 5231 or {str(p.relative_to(source)) for p in all_files} != expected:
        raise ValueError('Input does not match the original 5231-motion selection')
    urdf = ET.parse(REPO / 'soma_retargeter/assets/robotics/unitree/h2/desc/H2.urdf')
    names = [j.get('name') for j in urdf.findall('joint') if j.get('type') == 'revolute']
    if len(names) != 31:
        raise ValueError('Expected 31 H2 revolute joints')
    shards = [[] for _ in range(args.workers)]
    skipped, invalid = 0, []
    for i, src in enumerate(all_files):
        dst = output / src.relative_to(source).with_suffix('.csv')
        if dst.exists():
            ok, reason = validate_csv(dst, bvh_frames(src), names)
            if ok:
                skipped += 1
                continue
            invalid.append((str(dst), reason))
        shards[i % args.workers].append(src)
    print(f'Total=5231 validated_complete={skipped} pending={sum(map(len, shards))}', flush=True)
    print(f'Workers={args.workers} GPU={args.gpu} per_worker_batch={args.batch_size} '
          f'max_concurrent_motions={args.workers * args.batch_size}', flush=True)
    print('Pending per worker:', list(map(len, shards)), flush=True)
    if invalid:
        for path, reason in invalid:
            print(f'INVALID: {path}: {reason}', flush=True)
        raise RuntimeError('Preserve/move these incomplete CSVs out of csv/ before retrying; nothing overwritten')
    if not args.execute or not any(shards):
        print('No retarget started.', flush=True)
        return
    # Lock protects against two launchers starting after the process scan.
    with (args.config.parent / '.parallel_retarget.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run = Path(tempfile.mkdtemp(prefix='parallel_', dir=args.config.parent))
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=args.gpu, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1',
                   OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
        children = []
        try:
            for i, files in enumerate(shards):
                if not files:
                    continue
                shard_dir = run / f'worker{i:02d}'
                inputs = shard_dir / 'input_bvh'
                for src in files:
                    link = inputs / src.relative_to(source)
                    link.parent.mkdir(parents=True, exist_ok=True)
                    link.symlink_to(src.resolve())
                config = dict(cfg, import_folder=str(inputs), batch_size=args.batch_size)
                path = shard_dir / 'config.json'
                path.write_text(json.dumps(config, indent=2) + '\n')
                child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                                          '--worker', '--config', str(path)],
                                         cwd=REPO, env=env, start_new_session=True)
                children.append(child)
                print(f'[worker {i}] PID={child.pid} pending={len(files)}', flush=True)
            while any(p.poll() is None for p in children):
                failed = [(p.pid, p.returncode) for p in children if p.poll() not in (None, 0)]
                if failed:
                    raise RuntimeError(f'Worker failure {failed}; validated CSVs remain resumable')
                time.sleep(1)
            if any(p.returncode != 0 for p in children):
                raise RuntimeError('A worker failed; rerun plan to check resumable outputs')
        finally:
            stop_children(children)
        missing = [p for p in all_files if not (output / p.relative_to(source).with_suffix('.csv')).is_file()]
        if missing:
            raise RuntimeError(f'{len(missing)} outputs still missing')
        print('COMPLETE: 5231 outputs (validated existing + atomically validated new).', flush=True)


if __name__ == '__main__':
    main()
