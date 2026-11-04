"""BVH -> candidate CSV -> SQP/QP CSV -> independently checked training PKL."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
from .workspace import paths, environment, materialize, write_json, digest, verify


def invoke(root, command, extra_env=None):
    verify(root)
    env = environment(root)
    env.update(extra_env or {})
    return subprocess.run(command, cwd=paths(root)['bundle'] / 'code',
                          env=env, check=True)


def prepare(args):
    p = materialize(args.workspace, args.groot_root)
    command = [sys.executable, '-m', 'gear_sonic.scripts.run_h2_bvh_reference_batch', 'prepare',
               '--out', str(p['source']), '--source-selection', str(args.selection.resolve()),
               '--bvh-root', str(args.bvh_root.resolve()), '--g1-csv-root', str(args.g1_csv_root.resolve())]
    if args.limit:
        command += ['--limit', str(args.limit)]
    invoke(args.workspace, command)
    from .publish import bvh_timing
    rows = json.loads((p['source'] / 'manifest.json').read_text())['motions']
    frames = {}
    for row in rows:
        count, fps = bvh_timing(row['bvh'])
        if fps != 120:
            raise ValueError('Recorded production recipe requires native120Hz: ' + row['key'])
        frames[row['key']] = count
    write_json(p['bundle'] / 'length_inventory.json', dict(frames=frames))
    # length_inventory is not consumed by initial retarget verification; bind it separately.
    write_json(p['root'] / 'selection_receipt.json', dict(selection_sha256=digest(args.selection), frames=frames))


def retarget(args):
    p = paths(args.workspace)
    invoke(args.workspace, [sys.executable, '-m', 'gear_sonic.scripts.run_h2_bvh_gpu_staged', 'run',
                           '--out', str(p['source']), '--frame-chunk', str(args.frame_chunk),
                           '--iterations', str(args.iterations), '--cpu-workers', str(args.workers),
                           '--stop-after', args.stop_after],
           extra_env=dict(H2_SOMA_PYTHON=args.soma_python or sys.executable))


def plan(args):
    p = paths(args.workspace)
    if not (p['source'] / 'complete.json').exists():
        raise ValueError('Finish initial retarget + independent G1/H2 audit first')
    rows = []
    for file in sorted((p['source'] / 'reports').glob('*.json')):
        report = json.loads(file.read_text())
        if report['status'] == 'error':
            raise ValueError('Unresolved source error: ' + report['key'])
        if report['status'] == 'accepted':
            continue
        reasons = sorted({x['section'] for x in report['checks'] if not x['passed']})
        if not report['physical_and_margin_pass']:
            reasons.append('velocity:joint_limits_margin_or_native_velocity')
        rows.append(dict(key=report['key'], reasons=reasons))
    write_json(p['source'] / 'retry_plan.json', dict(motions=rows, original_gates_unchanged=True))
    print(json.dumps(dict(requires_refinement=len(rows))))


def optimize(args):
    p = paths(args.workspace)
    if not (p['source'] / 'retry_plan.json').exists():
        raise ValueError('Run plan first')
    script = p['foot'] / ('batch_reopt.py' if args.backend == 'osqp' else 'batch_reopt_interior.py')
    if args.backend == 'piqp':
        if not list((p['foot'] / 'vendor_piqp').rglob('*.so')):
            raise ValueError('Install piqp==0.6.4 into ' + str(p['foot'] / 'vendor_piqp'))
    command = [sys.executable, str(script), '--all-failed', '--out', str(p['optimized']),
               '--resume', '--workers', str(args.workers), '--iterations', str(args.iterations),
               '--refit-iterations', str(args.refit_iterations), '--foot-iterations', str(args.foot_iterations)]
    if args.dry_run:
        command.append('--dry-run')
    invoke(args.workspace, command)


def publish(args):
    from .publish import publish_workspace
    print(json.dumps(publish_workspace(args.workspace, args.out, args.smpl_root, args.soma_root)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='stage', required=True)
    for name in ['prepare', 'retarget', 'plan', 'optimize', 'publish']:
        p = sub.add_parser(name)
        p.add_argument('--workspace', type=Path, required=True)
        p.set_defaults(action=globals()[name])
        if name == 'prepare':
            for flag in ['groot-root', 'selection', 'bvh-root', 'g1-csv-root']:
                p.add_argument('--' + flag, type=Path, required=True)
            p.add_argument('--limit', type=int)
        if name in ['retarget', 'optimize']:
            p.add_argument('--workers', type=int, default=4)
            p.add_argument('--iterations', type=int, default=100 if name == 'retarget' else 30)
        if name == 'retarget':
            p.add_argument('--soma-python', help='Optional separate Newton/Warp interpreter for raw BVH IK')
            p.add_argument('--frame-chunk', type=int, default=8192)
            p.add_argument('--stop-after', choices=['raw', 'refine', 'audit'], default='audit')
        if name == 'optimize':
            p.add_argument('--backend', choices=['osqp', 'piqp'], default='osqp')
            p.add_argument('--refit-iterations', type=int, default=32)
            p.add_argument('--foot-iterations', type=int, default=24)
            p.add_argument('--dry-run', action='store_true')
        if name == 'publish':
            p.add_argument('--out', type=Path, required=True)
            p.add_argument('--smpl-root', type=Path)
            p.add_argument('--soma-root', type=Path)
    args = parser.parse_args()
    for name in ['workers', 'iterations', 'frame_chunk', 'refit_iterations', 'foot_iterations', 'limit']:
        value = getattr(args, name, None)
        if value is not None and value < 1:
            parser.error(name + ' must be positive')
    args.action(args)


if __name__ == '__main__':
    main()
