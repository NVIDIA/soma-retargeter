"""Review exact training PKLs beside source BVH; never rerun IK or write motions.

Run from repository root with PYTHONPATH=. and --viewer gl. Pickle/joblib inputs
must be trusted. Labels are manual visual judgements, not physics certification.
"""
import json
from pathlib import Path


def load_training_motion(path, key):
    import joblib
    import numpy as np
    record = joblib.load(path)[key]
    q = np.asarray(record['dof'])
    p = np.asarray(record['root_trans_offset'])
    r = np.asarray(record['root_rot'])  # xyzw; no Euler conversion or retargeting
    n = len(q)
    if n < 2 or q.shape != (n, 31) or p.shape != (n, 3) or r.shape != (n, 4):
        raise ValueError(f'Invalid H2 shapes: {key}')
    data = np.concatenate([p, r, q], axis=1).astype(np.float32)
    fps = float(record['fps'])
    if not np.isfinite(data).all() or not np.isfinite(fps) or fps <= 0:
        raise ValueError(f'Nonfinite motion / invalid FPS: {key}')
    if not np.allclose(np.linalg.norm(r, axis=1), 1, atol=1e-3):
        raise ValueError(f'Invalid root quaternion: {key}')
    return data, fps


def main():
    import newton
    import newton.examples
    import warp as wp
    from app.bvh_to_csv_converter import Viewer
    from soma_retargeter.assets.csv import H231DOF_CSVConfig
    from soma_retargeter.robotics.csv_animation_buffer import CSVAnimationBuffer

    parser = newton.examples.create_parser()
    parser.set_defaults(viewer='gl')
    parser.add_argument('--review-root', type=Path, required=True)
    parser.add_argument('--start', type=int, default=None, help='1-based; default first unreviewed')
    parser.add_argument('--contains', default='', help='Optional case-insensitive key filter')
    parser.add_argument('--focus-first', action='store_true')
    viewer, args = newton.examples.init(parser)
    if isinstance(viewer, newton.viewer.ViewerNull):
        raise ValueError('Use --viewer gl; this is a visual review, not a converter.')
    manifest = json.loads((args.review_root / 'review_manifest.json').read_text())
    items = [i for i in manifest['items'] if args.contains.lower() in i['key'].lower()]
    if args.focus_first:
        focus = set(manifest['roles']['focus_keys']) if 'focus_keys' in manifest['roles'] else set(manifest['roles']['focus_labels'])
        items.sort(key=lambda i: (i['key'] not in focus, i['key']))
    if not items:
        raise ValueError('No matching motions')
    for item in items:
        for field in ('pkl', 'bvh'):
            p = args.review_root / 'data' / item[field]
            if not p.is_file() or p.stat().st_size != item[field + '_bytes']:
                raise ValueError(f'Missing/incomplete download: {p}; rerun rsync')
    notes_path = args.review_root / 'review_labels.jsonl'
    labels = {}
    if notes_path.exists():
        for line in notes_path.read_text().splitlines():
            row = json.loads(line)
            labels[row['key']] = row['label']

    class ReviewViewer(Viewer):
        def __init__(self):
            self.index = (args.start - 1) if args.start is not None else next(
                (i for i, item in enumerate(items) if item['key'] not in labels), 0)
            if not 0 <= self.index < len(items):
                raise ValueError('--start out of range')
            self.review_ready = False
            super().__init__(viewer, dict(retarget_target='h2', retarget_source='soma',
                                         retarget_source_facing_direction='Mujoco'))
            expected = [s.removesuffix('_dof') for s in H231DOF_CSVConfig.csv_header[7:]]
            actual = [s.split('/')[-1] for s in self.model.joint_label][-31:]
            if self.robot_num_joint_q != 38 or actual != expected:
                raise ValueError(f'H2 model joint order mismatch: {actual}')
            self.show_gizmos = False
            self.review_ready = True
            self.load_current()

        def load_current(self):
            item = items[self.index]
            data, fps = load_training_motion(args.review_root / 'data' / item['pkl'], item['key'])
            self.robot_csv_animation_buffers[0] = CSVAnimationBuffer.create_from_raw_data(data, fps)
            self.load_bvh_file(args.review_root / 'data' / item['bvh'])
            # Constant display translation only: same root trajectories, side by side.
            self.robot_offsets[0] = wp.transform(wp.vec3(0, 1.1, 0), wp.quat_identity())
            self.animation_offsets[0] = wp.transform(wp.vec3(0, -1.1, 0), wp.quat_identity())
            self.playback_time = 0.0
            self.is_playing = True
            human = self.animation_buffers[0]
            self.timing_note = (f'PKL {len(data)} frames @ {fps:g} Hz | '
                                f'BVH {human.num_frames} frames @ {human.sample_rate:g} Hz')
            print(f'[{self.index + 1}/{len(items)}] {item["key"]}\n{self.timing_note}', flush=True)
            viewer.renderer.set_title(f'H2 training reference: {item["key"]}')

        def move(self, delta):
            self.index = max(0, min(len(items) - 1, self.index + delta))
            self.load_current()

        def mark(self, label):
            from datetime import datetime, timezone
            key = items[self.index]['key']
            with notes_path.open('a') as f:
                f.write(json.dumps(dict(key=key, label=label,
                                        utc=datetime.now(timezone.utc).isoformat())) + '\n')
            labels[key] = label
            self.move(1)

        def gui(self, ui):
            if not self.review_ready:
                return
            self.ui_playback_controls(ui)
            ui.begin('H2 training data review')
            ui.text(f'{self.index + 1}/{len(items)}  Reviewed: {sum(i["key"] in labels for i in items)}')
            ui.text_wrapped(items[self.index]['key'])
            ui.text_wrapped(self.timing_note)
            ui.text('Kinematic reference playback, NOT a policy rollout')
            ui.text('Side by side: original SOMA / actual training H2 PKL')
            ui.text('Recorded label: ' + labels.get(items[self.index]['key'], 'unreviewed'))
            if ui.button('Previous'):
                self.move(-1)
            ui.same_line()
            if ui.button('Next (skip)'):
                self.move(1)
            for text, label in [('Looks OK + next', 'visual_ok'), ('Suspect + next', 'suspect'),
                                ('Needs external support + next', 'external_support')]:
                if ui.button(text):
                    self.mark(label)
            _, self.show_skeleton_mesh = ui.checkbox('Source mesh', self.show_skeleton_mesh)
            _, self.show_skeleton = ui.checkbox('Source skeleton', self.show_skeleton)
            ui.end()

    with wp.ScopedDevice(args.device):
        ReviewViewer().run()


if __name__ == '__main__':
    main()
