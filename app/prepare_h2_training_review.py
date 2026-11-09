"""Read-only server inventory; write selective rsync lists locally. No transfer."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

DEFAULT_RUN = (
    "/home/arrc/lib/groot_wbc/experiments/h2_h100_newton/"
    "h2_b_stage7_from50k_keep2_focus64_adaptive1024_10k_env16384_seed42_v1"
)
REMOTE_SCAN = r'''
import json, sys
from pathlib import Path
root, run = map(Path, sys.argv[1:])
roles = json.loads((run / 'motion_roles.json').read_text())
keys = roles['catalog_keys']
assert len(keys) == len(set(keys)) == 5231, 'Unexpected candidate catalog'
assert all(k and '/' not in k and '\\' not in k and '\n' not in k and '\r' not in k for k in keys)
key_set = set(keys)
index = {}
for p in (root / 'bones_soma_uniform_bvh').rglob('*.bvh'):
    if p.stem in key_set:
        index.setdefault(p.stem, []).append(p)
items = []
for k in keys:
    pkl = root / 'h2_motion_lib' / (k + '.pkl')
    assert pkl.is_file(), f'Missing PKL: {k}'
    matches = index.get(k, [])
    assert len(matches) == 1, f'Missing/ambiguous BVH: {k}: {matches}'
    bvh = matches[0]
    items.append(dict(key=k, pkl=str(pkl.relative_to(root)),
                      bvh=str(bvh.relative_to(root)),
                      pkl_bytes=pkl.stat().st_size, bvh_bytes=bvh.stat().st_size))
print(json.dumps(dict(source_run=str(run), data_root=str(root), roles=roles, items=items)))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--host', default='arrc@10.107.1.12')
    parser.add_argument('--port', type=int, default=2200)
    parser.add_argument('--run', default=DEFAULT_RUN)
    parser.add_argument('--data-root', default='/home/arrc/lib/groot_wbc/data')
    args = parser.parse_args()
    command = shlex.join(['python3', '-', args.data_root, args.run])
    result = subprocess.run(['ssh', '-p', str(args.port), args.host, command],
                            input=REMOTE_SCAN, text=True, stdout=subprocess.PIPE, check=True)
    manifest = json.loads(result.stdout)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / 'review_manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise SystemExit('Existing manifest differs; choose a new output directory.')
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    (args.output / 'files.txt').write_text(''.join(
        p + '\n' for p in sorted({i[f] for i in manifest['items'] for f in ('pkl', 'bvh')})))
    total = sum(i['pkl_bytes'] + i['bvh_bytes'] for i in manifest['items'])
    print(f"{len(manifest['items'])} motion pairs, {total / 1e9:.2f} GB. No data transferred.")
    print(f'rsync list: {args.output / "files.txt"}')


if __name__ == '__main__':
    main()
