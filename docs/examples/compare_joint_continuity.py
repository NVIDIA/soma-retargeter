# SPDX-FileCopyrightText: Copyright (c) 2026 gohiroaki
# SPDX-License-Identifier: Apache-2.0
"""Run one licensed input BVH against bundled A3 with/without arm continuity."""
import argparse
import json
from pathlib import Path
import runpy
import shutil
import sys

from soma_retargeter.robotics.robot_registry import registry

p = argparse.ArgumentParser()
p.add_argument('--input', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--device', default='cuda:0')
a = p.parse_args()
target = registry.get('agibot-a3t3')
asset = Path(target.manifest_dir)
for variant, weight in [('baseline', 0), ('continuity', 2)]:
    root = (a.output/variant).resolve()
    if root.exists():
        raise FileExistsError(f'Refusing to overwrite {root}')
    robot = root/'robots/a3'
    shutil.copytree(asset/'configs', robot/'configs')
    (root/'input').mkdir(parents=True)
    shutil.copyfile(a.input, root/'input'/a.input.name)
    manifest = {'name': 'agibot-a3t3',
                'desc': {'urdf_path': str(asset/'desc/agibot-a3t3.urdf')},
                'retarget_configs': target.retarget_configs}
    (robot/'manifest.json').write_text(json.dumps(manifest, indent=2))
    path = robot/target.retarget_configs['soma']
    config = json.loads(path.read_text())
    if weight:
        config['joint_continuity_weights'] = {
            f'{side}_{joint}_joint': weight
            for side in ('left', 'right')
            for joint in ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow',
                          'wrist_roll', 'wrist_pitch', 'wrist_yaw')}
    path.write_text(json.dumps(config, indent=2))
    converter = {'import_folder': str(root/'input'), 'export_folder': str(root/'a3'),
                 'batch_size': 1, 'retargeter': 'Newton', 'retarget_source': 'soma',
                 'retarget_target': 'agibot-a3t3', 'retarget_source_facing_direction': 'Mujoco',
                 'extra_robot_paths': [str(root/'robots')]}
    (root/'a3.json').write_text(json.dumps(converter, indent=2))
    sys.argv = ['bvh_to_csv_converter', '--device', a.device, '--viewer', 'null',
                '--config', str(root/'a3.json')]
    runpy.run_module('app.bvh_to_csv_converter', run_name='__main__')
