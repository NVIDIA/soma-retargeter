"""Build an isolated workspace around the recorded production numerical code."""
from pathlib import Path
import hashlib
import json
import os
import shutil

PRODUCTION = Path(__file__).resolve().parent / 'production'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def paths(root):
    root = Path(root).resolve()
    return dict(root=root, bundle=root / 'artifacts/h2_bvh_gpu_staged_v5',
                source=root / 'artifacts/h2_bvh_gpu_refine131k_116014_v6',
                foot=root / 'artifacts/h2_foot_trajectory_v10',
                optimized=root / 'artifacts/h2_final_reopt')


def environment(root):
    p = paths(root)
    bundle = p['bundle']
    return dict(os.environ, H2_WORKSPACE=str(p['root']),
                PYTHONPATH=os.pathsep.join([str(bundle / 'code'), str(bundle / 'soma-retargeter-h2')]),
                SOMA_RETARGETER_ROOT=str(bundle / 'soma-retargeter-h2'),
                H2_MJCF_PATH=str(bundle / 'code/gear_sonic/data/assets/robot_description/mjcf/h2.xml'),
                OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')


def verify(root):
    p = paths(root)
    receipt = json.loads((p['root'] / 'workspace.json').read_text())
    if digest(p['bundle'] / 'bundle_manifest.json') != receipt['bundle_sha256']:
        raise ValueError('Workspace bundle manifest changed')
    manifest = json.loads((p['bundle'] / 'bundle_manifest.json').read_text())
    for relative, expected in manifest['files'].items():
        if digest(p['bundle'] / relative) != expected:
            raise ValueError('Sealed model/code/calibration changed: ' + relative)
    for relative, expected in receipt['runtime_files'].items():
        if digest(p['root'] / relative) != expected:
            raise ValueError('Sealed optimization code changed: ' + relative)
    return receipt


def materialize(root, groot_root):
    """Copy code/calibration, link caller-owned meshes, seal every runtime input.

    Only historical workspace literals change in copied numerical sources.
    Production source files, objectives and thresholds remain unchanged.
    """
    import numpy as np
    import xml.etree.ElementTree as ET
    p = paths(root)
    if p['root'].exists():
        raise FileExistsError('Use a NEW workspace: ' + str(p['root']))
    groot = Path(groot_root).resolve()
    assets = groot / 'gear_sonic/data/assets/robot_description'
    if not (assets / 'urdf/h2/meshes').is_dir():
        raise FileNotFoundError('H2 meshes are required under ' + str(assets))
    p['root'].mkdir(parents=True)
    bundle = p['bundle']
    shutil.copytree(PRODUCTION / 'code', bundle / 'code')
    for folder in [bundle / 'code/gear_sonic', bundle / 'code/gear_sonic/scripts']:
        (folder / '__init__.py').touch(exist_ok=True)
    shutil.copytree(PRODUCTION / 'soma_retargeter', bundle / 'soma-retargeter-h2/soma_retargeter')
    somaconfig = bundle / 'soma-retargeter-h2/soma_retargeter/configs/soma'
    for name in ['soma_zero_frame0.bvh', 'soma_base_skel_minimal.usd']:
        source = Path(__file__).resolve().parents[1] / 'soma_retargeter/configs/soma' / name
        if source.read_bytes().startswith(b'version https://git-lfs.github.com'):
            raise ValueError('Run git lfs pull first: ' + str(source))
        shutil.copy2(source, somaconfig / name)
    model = bundle / 'code/gear_sonic/data/assets/robot_description'
    shutil.copytree(PRODUCTION / 'models', model)
    (model / 'meshes').symlink_to(assets / 'meshes', target_is_directory=True)
    (model / 'urdf/h2/meshes').symlink_to(assets / 'urdf/h2/meshes', target_is_directory=True)
    shutil.copytree(PRODUCTION / 'profile', bundle / 'profile')
    raw = json.loads((bundle / 'profile/reference.json').read_text())
    np.savez_compressed(bundle / 'profile/reference.npz',
                        **{k: np.array(v['value'], dtype=v['dtype']) for k, v in raw.items()})
    config = json.loads((bundle / 'profile/retarget.json').read_text())
    config.update(human_robot_scaler_config=str(bundle / 'profile/scaler.json'),
                  feet_stabilizer_config=str(bundle / 'profile/feet.json'))
    write_json(bundle / 'profile/retarget.json', config)
    write_json(bundle / 'approved/manifest.json', {})
    shutil.copytree(PRODUCTION / 'foot', p['foot'])
    scripts = p['root'] / 'workspace/GR00T-WholeBodyControl-h2-clean/gear_sonic/scripts'
    shutil.copytree(PRODUCTION / 'code/gear_sonic/scripts', scripts)
    relocations = []
    for folder in [p['foot'], scripts, bundle / 'code']:
        for source in folder.rglob('*.py'):
            text = source.read_text()
            original = digest(source)
            # This is a filesystem relocation only, not a numerical substitution.
            text = text.replace("Path('/home/arrc/lib/groot_wbc')", 'Path(__import__("os").environ["H2_WORKSPACE"])')
            text = text.replace("default='/home/arrc/.conda/envs/soma-retargeter/bin/python'",
                                "default=os.environ.get('H2_SOMA_PYTHON',sys.executable)")
            text = text.replace("default='/home/arrc/.conda/envs/groot-wbc/bin/python'", "default=sys.executable")
            source.write_text(text)
            if digest(source) != original:
                relocations.append(dict(path=str(source.relative_to(p['root'])), original_sha256=original,
                                        relocated_sha256=digest(source)))
    write_json(p['foot'] / 'unresolved_progress.json', dict(motions={}))
    for name in ['h2_accepted52219_compound_v9', 'h2_accepted52217_recovery_v6']:
        write_json(p['root'] / 'artifacts' / name / 'catalog.json', [])
    inputs = {str(x.relative_to(bundle)): digest(x) for x in bundle.rglob('*') if x.is_file()}
    # rglob does not descend into linked mesh directories. Seal referenced meshes explicitly.
    for xml in (model / 'mjcf').glob('*.xml'):
        tree = ET.parse(xml).getroot()
        directory = xml.parent / tree.find('compiler').get('meshdir', '')
        for item in tree.findall('asset/mesh'):
            mesh = directory / item.get('file')
            if mesh.read_bytes().startswith(b'version https://git-lfs.github.com'):
                raise ValueError('Unfetched mesh LFS pointer: ' + str(mesh))
            inputs[str(mesh.relative_to(bundle))] = digest(mesh)
    write_json(bundle / 'bundle_manifest.json', dict(schema='h2_repro_bundle.v1', files=inputs))
    runtime_files = {str(x.relative_to(p['root'])): digest(x)
                     for folder in [p['foot'], scripts] for x in folder.rglob('*.py')}
    write_json(p['root'] / 'workspace.json', dict(schema='h2_pipeline_workspace.v1', groot_root=str(groot),
               bundle_sha256=digest(bundle / 'bundle_manifest.json'), source_relocations=relocations,
               runtime_files=runtime_files, numerical_objectives_changed=False, source_units='native_m_rad_120hz'))
    return p
