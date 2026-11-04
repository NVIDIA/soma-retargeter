"""Pack admission-passing native CSVs; optimization happens before this boundary."""
from pathlib import Path
import csv
import importlib.util
import json
import os
import tempfile
import xml.etree.ElementTree as ET
import joblib
import numpy as np
from scipy.spatial.transform import Rotation
from .workspace import paths, digest, write_json, verify

ROOT_COLUMNS = ['Frame', 'root_translateX', 'root_translateY', 'root_translateZ',
                'root_rotateX', 'root_rotateY', 'root_rotateZ']


def bvh_timing(path):
    frames = dt = None
    with Path(path).open() as stream:
        for line in stream:
            if line.strip().startswith('Frames:'):
                frames = int(line.split(':', 1)[1])
            elif line.strip().startswith('Frame Time:'):
                dt = float(line.split(':', 1)[1]); break
    if frames is None or frames < 2 or dt is None or not np.isfinite(dt) or dt <= 0:
        raise ValueError('Invalid BVH timing: ' + str(path))
    fps = round(1 / dt)
    if abs(1 / dt - fps) > .01:
        raise ValueError('Noninteger BVH FPS requires an explicit resampling contract')
    return frames, fps


def joint_contract(xml):
    bodies = list(ET.parse(xml).getroot().find('worldbody').iter('body'))
    joints = []
    for row, body in enumerate(bodies):
        for joint in body.findall('joint'):
            if joint.get('type', 'hinge') == 'free':
                continue
            if joint.get('type', 'hinge') != 'hinge':
                raise ValueError('Expected only31 scalar hinges and a free root')
            axis = np.fromstring(joint.get('axis', '0 0 1'), sep=' ')
            if axis.shape != (3,) or not np.isclose(np.linalg.norm(axis), 1):
                raise ValueError('Invalid hinge axis')
            joints.append((joint.get('name'), row, axis))
    if len(joints) != 31 or len(bodies) != 32 or len({x[0] for x in joints}) != 31:
        raise ValueError('Expected canonical H2 31-joint/32-body geometry')
    return joints


def convert_csv(csv_path, bvh, xml, *, units):
    if units not in ['native-m-rad', 'official-cm-deg']:
        raise ValueError('CSV units must be explicit')
    with Path(csv_path).open() as stream:
        reader = csv.reader(stream); header = next(reader); values = np.array(list(reader), float)
    clean = [name.removesuffix('_dof') for name in header]
    joints = joint_contract(xml); names = [x[0] for x in joints]
    if clean[:7] != ROOT_COLUMNS or len(clean) != 38 or set(clean[7:]) != set(names) or len(set(clean)) != 38:
        raise ValueError('Invalid or duplicate named H2 CSV columns')
    frames, fps = bvh_timing(bvh)
    if values.shape != (frames, 38) or not np.isfinite(values).all() or not np.array_equal(values[:, 0], np.arange(frames)):
        raise ValueError('CSV/BVH timing, frame ID or finite-value mismatch')
    q = values[:, [clean.index(name) for name in names]]
    root = values[:, 1:4]
    if units == 'official-cm-deg':
        q = np.deg2rad(q); root = root * .01
    rotation = Rotation.from_euler('xyz', values[:, 4:7], degrees=units == 'official-cm-deg')
    pose = np.zeros((frames, 32, 3), np.float32); pose[:, 0] = rotation.as_rotvec()
    dof = q.astype(np.float32)
    for column, (_, row, axis) in enumerate(joints):
        pose[:, row] = dof[:, column, None] * axis
    payload = dict(root_trans_offset=root.astype(np.float32), root_rot=rotation.as_quat().astype(np.float32),
                   dof=dof, pose_aa=pose, smpl_joints=np.zeros((frames, 24, 3), np.float32), fps=fps)
    return payload, header, values, names


def certificate(report, csv_path):
    if digest(csv_path) != report['refined_sha256'] or not report['physical_and_margin_pass']:
        raise ValueError('CSV certificate/hash or native physical checks failed')
    batch = report.get('batch_validation')
    if batch is not None:
        if batch['status'] not in ['legacy_accepted', 'new_H2_reference_validated'] or not batch['physical_pass']:
            raise ValueError('Refinement is still under review')
        if not all(v['penetrating_samples'] == 0 and v['grounded_target_above_0p5mm'] == 0 for v in batch['feet'].values()):
            raise ValueError('Refinement foot gates failed')
        if batch['status'] == 'new_H2_reference_validated':
            if not (batch['new_reference_pass'] and batch['fixed_reference']['passed']
                    and batch['raw_quality_review']['primary_nonregression_pass']):
                raise ValueError('New H2 reference/fidelity/nonregression certificate failed')
    elif not (report['status'] == 'accepted' and report['checks'] and all(x['passed'] for x in report['checks'])):
        raise ValueError('Original G1/H2 comparison has not passed')


def validate_native(payload, model, urdf):
    names = [model.joint(i).name for i in range(1, model.njnt)]
    q = payload['dof'].astype(float)
    caps = {j.get('name'): float(j.find('limit').get('velocity'))
            for j in ET.parse(urdf).getroot().findall('joint') if j.find('limit') is not None}
    ratio = float((abs(np.diff(q, axis=0)) * payload['fps'] / np.array([caps[n] for n in names])).max())
    excess = float(np.maximum(np.maximum(model.jnt_range[1:, 0] - q, q - model.jnt_range[1:, 1]), 0).max())
    arm = [i for i, n in enumerate(names) if 'shoulder' in n or 'wrist' in n]
    margin = float(np.minimum(q[:, arm] - model.jnt_range[1 + np.array(arm), 0],
                              model.jnt_range[1 + np.array(arm), 1] - q[:, arm]).min())
    if excess > np.deg2rad(1e-4) or ratio > 1 + 1e-6 or margin < np.deg2rad(2 - 1e-5):
        raise ValueError(f'Float32 PKL joint limits/velocity/margins failed: {excess=}, {ratio=}, {margin=}')
    return dict(joint_limit_excess_rad=excess, max_native_velocity_ratio=ratio,
                shoulder_wrist_min_margin_deg=float(np.rad2deg(margin)))


def dense_foot_check(payload, header, values, names, trace, workspace):
    """Rebuild actual float32 PKL poses; use the frozen independent BVH evaluator."""
    p = paths(workspace); path = p['foot'] / 'audit_bvh_reconstruction.py'
    spec = importlib.util.spec_from_file_location('h2_independent_export_audit', path)
    module = importlib.util.module_from_spec(spec)
    previous = os.environ.get('H2_WORKSPACE')
    os.environ['H2_WORKSPACE'] = str(p['root'])
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            os.environ.pop('H2_WORKSPACE', None)
        else:
            os.environ['H2_WORKSPACE'] = previous
    module.BUNDLE = p['bundle']
    actual = values.copy(); clean = [x.removesuffix('_dof') for x in header]
    actual[:, 1:4] = payload['root_trans_offset']
    actual[:, 4:7] = Rotation.from_quat(payload['root_rot']).as_euler('xyz')
    actual[:, [clean.index(name) for name in names]] = payload['dof']
    with tempfile.TemporaryDirectory(prefix='h2_pkl_audit_') as directory:
        candidate = Path(directory) / 'float32.csv'
        with candidate.open('w') as stream:
            writer = csv.writer(stream); writer.writerow(header); writer.writerows(actual)
        evaluation = module.Evaluation('h2', candidate, trace)
        rates = {str(rate): evaluation.evaluate(rate) for rate in [120, 50, 600]}
    if any(v['penetrating_samples'] or v['grounded_target_above_0p5mm'] for v in rates.values()):
        raise ValueError('Actual float32 PKL fails120/50/600Hz BVH-grounded foot gates')
    return rates


def unique_index(root):
    if root is None:
        return {}
    result = {}
    for path in sorted(Path(root).rglob('*.pkl')):
        if path.stem in result:
            raise ValueError('Ambiguous paired basename: ' + path.stem)
        result[path.stem] = path.resolve()
    return result


def publish_workspace(workspace, output, smpl_root=None, soma_root=None):
    import mujoco
    p = paths(workspace); output = Path(output).resolve()
    verify(workspace)
    if output.exists():
        raise FileExistsError('Use a NEW PKL output directory')
    modelroot = p['bundle'] / 'code/gear_sonic/data/assets/robot_description'
    xml = modelroot / 'mjcf/h2.xml'; urdf = modelroot / 'urdf/h2/h2.urdf'
    model = mujoco.MjModel.from_xml_path(str(xml))
    source = {r['key']: r for r in json.loads((p['source'] / 'manifest.json').read_text())['motions']}
    jobs = {}
    for jobfile in (p['source'] / 'work').glob('gpu*_batch*/jobs.json'):
        for job in json.loads(jobfile.read_text()):
            if job['key'] in jobs:
                raise ValueError('Duplicate raw reference job: ' + job['key'])
            jobs[job['key']] = job
    candidates = {f.stem: (p['source'] / 'refined/csv' / (f.stem + '.csv'), f)
                  for f in (p['source'] / 'reports').glob('*.json')}
    for f in (p['optimized'] / 'validated/reports').glob('*.json'):
        candidates[f.stem] = (p['optimized'] / 'validated/csv' / (f.stem + '.csv'), f)
    paired = dict(smpl=unique_index(smpl_root), soma=unique_index(soma_root))
    (output / 'motion').mkdir(parents=True); accepted = []; rejected = []
    for key, (csv_path, report_path) in sorted(candidates.items()):
        try:
            report = json.loads(report_path.read_text()); certificate(report, csv_path)
            job = jobs[key]; item = source[key]
            if len(job['raw_files']) != 4 or report['raw_files'] != job['raw_files']:
                raise ValueError('Independent report/raw trace provenance mismatch')
            if digest(item['bvh']) != job['bvh_sha256'] or report['source_bvh_sha256'] != job['bvh_sha256']:
                raise ValueError('Original BVH identity mismatch')
            for relative, expected in job['raw_files'].items():
                if digest(Path(job['work']) / relative) != expected:
                    raise ValueError('Raw BVH/G1 trace identity changed: ' + relative)
            payload, header, values, names = convert_csv(csv_path, item['bvh'], xml, units='native-m-rad')
            if payload['fps'] != 120:
                raise ValueError('This refinement recipe requires native120Hz')
            checks = validate_native(payload, model, urdf)
            feet = dense_foot_check(payload, header, values, names,
                                    Path(job['work']) / 'h2' / (key + '_trace.npz'), workspace)
            pairs = {}
            for kind, index in paired.items():
                if index:
                    pair = index[key]
                    pairs[kind] = dict(path=str(pair), sha256=digest(pair))
            # Float32 serialization itself is checked before any public output.
            target = output / 'motion' / (key + '.pkl'); temporary = target.with_suffix('.partial')
            joblib.dump({key: payload}, temporary, compress=1)
            actual = joblib.load(temporary)[key]
            for name, value in payload.items():
                np.testing.assert_array_equal(actual[name], value)
            os.replace(temporary, target)
            record = dict(key=key, path=str(target), pkl_sha256=digest(target), frames=len(payload['dof']), fps=120,
                          csv_sha256=digest(csv_path), report_sha256=digest(report_path), bvh_sha256=job['bvh_sha256'],
                          paired=pairs, native_checks=checks, foot_checks=feet,
                          admission_kind=report.get('batch_validation', {}).get('status', 'legacy_accepted'),
                          scope='Kinematic export only; no training-pool mutation or dynamics/hardware claim')
            accepted.append(record); write_json(output / 'reports' / (key + '.json'), record)
        except Exception as exc:
            rejected.append(dict(key=key, reason=type(exc).__name__ + ': ' + str(exc)))
    write_json(output / 'catalog.json', accepted)
    manifest = dict(schema='h2_native_csv_pkl_handoff.v1', accepted=len(accepted), requires_review=len(rejected),
                    csv_units='native-m-rad', root_quaternion='xyzw', native_fps=120,
                    model_sha256=digest(xml), urdf_sha256=digest(urdf),
                    catalog_sha256=digest(output / 'catalog.json'), training_started=False)
    write_json(output / 'manifest.json', manifest); write_json(output / 'requires_review.json', rejected)
    return manifest
