"""Isolated ten-clip H2 retarget ablations. Never changes production assets."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys

BASE = Path('/home/arrc/lib/groot_wbc')
H2 = BASE/'workspace/GR00T-WholeBodyControl-h2-clean'
REPO = Path(__file__).resolve().parents[1]


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def prepare(root):
    if root.exists():
        raise FileExistsError(root)
    rows = [json.loads(p.read_text()) for p in (BASE/'artifacts/h2_retarget_old_new_audit_v1/clips').glob('*.json')]
    rows = [r for r in rows if r['new']['foot_box_min_z_cm'] >= -10
            and not any(s in r['key'] for s in ('postmortem', 'crawl', 'fall_'))]
    chosen = sorted(rows, key=lambda r: (-r['new']['max_leg_jump_deg_50hz'], r['key']))[:10]
    source = json.loads((BASE/'artifacts/h2_soma_v02_5231_bdAmYb/selection.json').read_text())
    index = {m['key']: m for m in source['motions']}
    motions = [index[r['key']] for r in chosen]
    root.mkdir(parents=True)
    save(root/'selection.json', dict(motions=motions, selection_rule='Top10 leg jumps; exclude foot box <-10cm, postmortem/crawl/fall labels', baseline_audit=chosen))
    original = REPO/'soma_retargeter/assets/robotics/unitree/h2/configs'
    config = json.loads((original/'soma_to_h2_retargeter_config.json').read_text())
    post = json.loads((original/'h2_post_processing_config.json').read_text())
    save(root/'base_config.json', config); save(root/'base_post.json', post)
    save(root/'provenance.json', dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), motions=10))
    print(json.dumps([r['key'] for r in chosen], indent=2), flush=True)


def variant_config(root, name):
    cfg = json.loads((root/'base_config.json').read_text())
    post = json.loads((root/'base_post.json').read_text())
    if name == 'base_only':
        cfg['enable_post_processing'] = False
        cfg['enable_contact_processing'] = False
    elif name == 'contact_off':
        cfg['enable_contact_processing'] = False
    elif name == 'contact_no_flatten':
        post['contact_correction']['enable_flatten_foot_plant'] = False
    elif name == 'post_orientation_low':
        for side in ('left','right'):
            for link in ('hip_roll_link','knee_link','ankle_pitch_link'):
                post['limb_stabilizer']['effectors'][side+'_'+link][1] *= 0.1
    elif name == 'base_footrot_low':
        cfg['enable_post_processing'] = False
        cfg['enable_contact_processing'] = False
        for part in ('LeftFoot','RightFoot'):
            cfg['ik_match_table'][part]['r_weight'] *= 0.1
    elif name in ('lowrot_softpost', 'lowrot_softpost_no_contact', 'base_legprior', 'legprior_softpost'):
        cfg['joint_limit_weight'] = 50.0
        cfg['ik_iterations'] = 48
        for part in ('LeftFoot','RightFoot'):
            cfg['ik_match_table'][part]['r_weight'] *= 0.1
        for side in ('left','right'):
            post['limb_stabilizer']['effectors'][side+'_ankle_pitch_link'] = [5.0, 0.2]
            post['limb_stabilizer']['effectors'][side+'_knee_link'] = [4.0, 0.2]
        if name in ('base_legprior', 'legprior_softpost'):
            for side in ('left','right'):
                for link in ('hip_pitch_link','hip_roll_link','hip_yaw_link','knee_link','ankle_pitch_link','ankle_roll_link'):
                    cfg['smooth_joint_filter_objective_body_masks'][side+'_'+link] = [0.3, 0.1, 0.1]
        if name == 'base_legprior':
            cfg['enable_post_processing'] = False
            cfg['enable_contact_processing'] = False
        if name == 'lowrot_softpost_no_contact':
            cfg['enable_contact_processing'] = False
    elif name in ('base_footrot_03', 'base_footrot_003', 'base_legrot_low', 'lowrot_contact_slow'):
        cfg['enable_post_processing'] = name == 'lowrot_contact_slow'
        cfg['enable_contact_processing'] = name == 'lowrot_contact_slow'
        ratio = {'base_footrot_03': 0.3, 'base_footrot_003': 0.03}.get(name, 0.1)
        for part in ('LeftFoot','RightFoot'):
            cfg['ik_match_table'][part]['r_weight'] *= ratio
        if name == 'base_legrot_low':
            for part in ('LeftLeg','RightLeg','LeftShin','RightShin'):
                cfg['ik_match_table'][part]['r_weight'] *= 0.1
        if name == 'lowrot_contact_slow':
            post['contact_correction']['transition_frames'] = 24
            post['contact_correction']['propagation_ratio'] = 0.1
            post['contact_correction']['rotation_propagation_ratio'] = 0.0
            for side in ('left','right'):
                post['limb_stabilizer']['effectors'][side+'_ankle_pitch_link'] = [5.0, 0.2]
                post['limb_stabilizer']['effectors'][side+'_knee_link'] = [4.0, 0.2]
    elif name == 'balanced_v2':
        cfg['joint_limit_weight'] = 50.0
        post['limb_stabilizer']['joint_limit_weight'] = 50.0
        post['contact_correction']['enable_flatten_foot_plant'] = False
        for part in ('LeftFoot','RightFoot'):
            cfg['ik_match_table'][part]['r_weight'] *= 0.1
        for side in ('left','right'):
            for link in ('hip_roll_link','knee_link','ankle_pitch_link'):
                post['limb_stabilizer']['effectors'][side+'_'+link][1] *= 0.1
    elif name == 'post_limit100':
        post['limb_stabilizer']['joint_limit_weight'] = 100.0
    elif name == 'post_softfoot':
        for side in ('left','right'):
            post['limb_stabilizer']['effectors'][side+'_ankle_pitch_link'] = [5.0, 0.2]
            post['limb_stabilizer']['effectors'][side+'_knee_link'] = [4.0, 2.0]
    elif name == 'post_combined':
        post['limb_stabilizer']['joint_limit_weight'] = 100.0
        post['limb_stabilizer']['ik_iterations'] = 40
        post['limb_stabilizer']['root_correction_ratio'] = 0.5
        for side in ('left','right'):
            post['limb_stabilizer']['effectors'][side+'_ankle_pitch_link'] = [5.0, 0.2]
            post['limb_stabilizer']['effectors'][side+'_knee_link'] = [4.0, 2.0]
    elif name != 'baseline':
        raise ValueError(name)
    return cfg, post


def run(root, name):
    import numpy as np
    import warp as wp
    import newton
    from soma_retargeter.io import bvh, csv
    from soma_retargeter.pipelines.soma_retargeting_pipeline import SomaRetargetingPipeline
    from soma_retargeter.pipelines import ik_pass
    from soma_retargeter.utils import newton_utils
    from soma_retargeter.utils.space_conversion_utils import SpaceConverter, get_facing_direction_type_from_str
    motions = json.loads((root/'selection.json').read_text())['motions']
    out = root/name
    out.mkdir(exist_ok=False)
    cfg, post = variant_config(root, name)
    save(out/'post.json', post)
    cfg['post_processing']['robot_config'] = str(out/'post.json')
    save(out/'config.json', cfg)
    source = BASE/'data/bones_soma_uniform_bvh'
    paths = [source/m['bvh'] for m in motions]
    with wp.ScopedDevice('cuda:0'):
        skel, _ = bvh.BVHImporter().create_skeleton(paths[0])
        animations = [bvh.load_bvh(p, skel)[1] for p in paths]
        pipeline = SomaRetargetingPipeline(skel, 'soma', 'unitree_h2', retarget_config=cfg)
        names = newton_utils.get_filtered_joint_names(pipeline.robot_builder,[newton.JointType.REVOLUTE])
        offset = SpaceConverter(get_facing_direction_type_from_str('Mujoco')).transform(wp.transform_identity())
        pipeline.add_input_motions(animations,[offset]*len(animations),True)
        trim = pipeline.num_initialization_frames + pipeline.num_stabilization_frames
        for i, m in enumerate(motions):
            np.savez_compressed(out/(m['key']+'_targets.npz'), targets=np.asarray(pipeline.input_targets[i][trim:]), names=np.array(pipeline.mapped_joints))
        trace = {'base': [], 'post': []}
        if name == 'baseline':
            # Diagnostic only: read each pass output. No solver state is changed.
            for klass, stage in [(ik_pass.BasePass, 'base'),(ik_pass.PostProcessPass, 'post')]:
                original = klass.execute
                def wrapped(self, frame, original=original, stage=stage):
                    result = original(self, frame)
                    if frame >= trim:
                        trace[stage].append(self._joint_q.numpy().copy())
                    return result
                klass.execute = wrapped
        buffers = pipeline.execute()
        for i, m in enumerate(motions):
            csv.save_csv(str(out/(m['key']+'.csv')), names, buffers[i])
        if name == 'baseline':
            np.savez_compressed(out/'pass_trace.npz', **{k:np.stack(v) for k,v in trace.items()}, names=np.array(names))
        save(out/'complete.json', dict(motions=len(buffers), variant=name))
        print('COMPLETE', name, flush=True)


def urdf_fk(path, names, q, root_pos, root_rot):
    import numpy as np
    import xml.etree.ElementTree as ET
    from scipy.spatial.transform import Rotation
    xml = ET.parse(path).getroot()
    pending = list(xml.findall('joint'))
    xyz = {'pelvis':root_pos}; rot = {'pelvis':root_rot}
    while pending:
        progress = False
        for j in pending[:]:
            parent, child = j.find('parent').get('link'), j.find('child').get('link')
            if parent not in xyz:
                continue
            origin = j.find('origin')
            shift = np.fromstring(origin.get('xyz','0 0 0') if origin is not None else '0 0 0',sep=' ')
            rpy = np.fromstring(origin.get('rpy','0 0 0') if origin is not None else '0 0 0',sep=' ')
            r = Rotation.from_euler('xyz',rpy).as_matrix()
            if j.get('type') == 'revolute':
                axis = np.fromstring(j.find('axis').get('xyz'),sep=' ')
                r = r @ Rotation.from_rotvec(q[:, names.index(j.get('name')), None]*axis).as_matrix()
            xyz[child] = xyz[parent] + np.einsum('tij,j->ti',rot[parent],shift)
            rot[child] = rot[parent] @ r
            pending.remove(j); progress = True
        if not progress:
            raise ValueError('Disconnected URDF')
    return xyz,rot


def score(root, name):
    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation
    from omegaconf import OmegaConf
    sys.path.insert(0,str(H2))
    from gear_sonic.data_process.convert_h2_soma_v02 import convert, joint_contract
    from gear_sonic.scripts.audit_h2_retarget_pair import screen
    from gear_sonic.utils.motion_lib.torch_humanoid_batch import Humanoid_Batch
    from gear_sonic.newton.asset_manifest import ANALYTIC_FOOT_BOXES
    import itertools
    torch.set_num_threads(2)
    run = BASE/'experiments/h2_h100_newton/h2_b_stage7_from50k_keep2_focus64_adaptive1024_10k_env16384_seed42_v1'
    cfg = OmegaConf.load(run/'config.yaml').manager_env.commands.motion.motion_lib_cfg
    cfg.asset.assetRoot = str(H2/'gear_sonic/data/assets/robot_description/mjcf');cfg.load_meshes=False
    fk=Humanoid_Batch(cfg); xml=Path(cfg.asset.assetRoot)/str(cfg.asset.assetFileName)
    jc=joint_contract(xml);names=[j[0] for j in jc]
    signs=np.array(list(itertools.product([-1,1],repeat=3)))
    feet=[(fk.body_names.index(n),np.array(b['translation'])+signs*np.array(b['size'])/2) for n,b in ANALYTIC_FOOT_BOXES.items()]
    out=root/name; motion_items=json.loads((root/'selection.json').read_text())['motions']
    conf=json.loads((out/'config.json').read_text());rows=[]
    for m in motion_items:
        record=convert(out/(m['key']+'.csv'),BASE/'data/bones_soma_uniform_bvh'/m['bvh'],jc)
        row=screen(record,fk,feet,fk.joints_range.numpy()); row['key']=m['key']
        target=np.load(out/(m['key']+'_targets.npz'));targets=target['targets'];t_names=target['names'].tolist()
        xyz,rot=urdf_fk(REPO/'soma_retargeter/assets/robotics/unitree/h2/desc/H2.urdf',names,record['dof'],record['root_trans_offset'],Rotation.from_quat(record['root_rot']).as_matrix())
        pos=[];ang=[]
        for i,n in enumerate(t_names):
            spec=conf['ik_match_table'][n]
            pos.append(np.linalg.norm(xyz[spec['t_body']]-targets[:,i,:3],axis=1))
            diff=Rotation.from_matrix(rot[spec['r_body']]).inv()*Rotation.from_quat(targets[:,i,3:])
            ang.append(diff.magnitude())
        row['target_pos_rmse_m']=float(np.sqrt(np.mean(np.square(pos))))
        row['target_rotation_rmse_deg']=float(np.rad2deg(np.sqrt(np.mean(np.square(ang)))))
        row['foot_target_pos_rmse_m']=float(np.sqrt(np.mean(np.square([pos[t_names.index(n)] for n in ('LeftFoot','RightFoot')]))))
        row['raw_leg_max_jump_deg']=float(np.rad2deg(np.abs(np.diff(record['dof'][:,:12],axis=0)).max()))
        rows.append(row)
    save(out/'scores.json',rows)
    summary=dict(variant=name, leg_jump_flagged=sum(r['flags']['leg_jump_over_20deg_per_20ms'] for r in rows), max_leg_jump=max(r['max_leg_jump_deg_50hz'] for r in rows), target_pos_rmse_mean=float(np.mean([r['target_pos_rmse_m'] for r in rows])), target_rot_rmse_mean=float(np.mean([r['target_rotation_rmse_deg'] for r in rows])), foot_target_rmse_mean=float(np.mean([r['foot_target_pos_rmse_m'] for r in rows])), worst_box_cm=min(r['foot_box_min_z_cm'] for r in rows), penetration_fraction_mean=float(np.mean([r['foot_box_below_3cm_fraction'] for r in rows])))
    save(out/'summary.json',summary);print(json.dumps(summary),flush=True)


def summarize(root):
    """Collect measured results only; never auto-promote a production config."""
    results = [json.loads(p.read_text()) for p in sorted(root.glob('*/summary.json'))]
    incomplete = [p.parent.name for p in root.glob('*/config.json')
                  if not (p.parent/'summary.json').is_file()]
    report = dict(
        completed_variants=len(results), incomplete_or_failed_variants=sorted(incomplete),
        results=results, production_candidate=None,
        all_ten_below_jump_screen=[s['variant'] for s in results if s['leg_jump_flagged']==0],
        note='Kinematic ten-clip tuning only, not policy evaluation or dynamics validation. 20deg/20ms is a diagnostic threshold, not an actuator limit.',
        final_runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    save(root/'comparison.json', report)
    print('| Variant | Flagged /10 | Max leg delta deg/20ms | Mean target RMSE cm | Mean foot RMSE cm | Worst foot box z cm | Mean fraction below -3cm |')
    print('|---|---:|---:|---:|---:|---:|---:|')
    for s in results:
        print(f"| {s['variant']} | {s['leg_jump_flagged']} | {s['max_leg_jump']:.2f} | {100*s['target_pos_rmse_mean']:.2f} | {100*s['foot_target_rmse_mean']:.2f} | {s['worst_box_cm']:.2f} | {s['penetration_fraction_mean']:.4f} |")
    print('Incomplete/failed:', incomplete)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run','score','summarize']);p.add_argument('--root',type=Path,required=True);p.add_argument('--variant',default='baseline');a=p.parse_args()
    if a.mode=='prepare':prepare(a.root.resolve())
    elif a.mode=='run':run(a.root.resolve(),a.variant)
    elif a.mode=='summarize':summarize(a.root.resolve())
    else:score(a.root.resolve(),a.variant)
