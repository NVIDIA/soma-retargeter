"""GPU1-only BVH-target retry with unchanged foot trajectory admission.

Default scope: 48 velocity-repair failures plus 8 unchanged audit controls.
No training-pool mutation, no G1 retarget, no CPU optimization fallback.
"""
import argparse
import concurrent.futures
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time


def read(p):return json.loads(Path(p).read_text())


def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def save(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,allow_nan=False)+'\n');tmp.replace(p)


def api_for(bundle):
    sys.path[:0]=[str(bundle/'code'),str(bundle/'soma-retargeter-h2')]
    os.environ['SOMA_RETARGETER_ROOT']=str(bundle/'soma-retargeter-h2')
    from gear_sonic.scripts import run_h2_bvh_gpu_staged as api
    if api.BUNDLE.resolve()!=bundle.resolve():raise ValueError('Wrong frozen bundle')
    return api


def trajectory_guard(model, baseline, candidate, fps):
    import csv
    import mujoco
    import numpy as np
    from scipy.spatial.transform import Rotation as R,Slerp
    def load(p):
        with Path(p).open() as f:
            reader=csv.reader(f);h=[n.removesuffix('_dof') for n in next(reader)];v=np.array(list(reader),float)
        return h,v
    header,a=load(baseline);hh,b=load(candidate)
    if hh!=header or a.shape!=b.shape or not np.array_equal(a[:,0],b[:,0]):raise ValueError('Changed timebase/schema')
    joints=[model.joint(i).name for i in range(1,model.njnt)];cols=[header.index(n) for n in joints]
    protected=[header.index(n) for n in joints if not any(k in n for k in ('hip','knee','ankle'))]
    if not np.array_equal(a[:,[0,4,5,6]+protected],b[:,[0,4,5,6]+protected]):raise ValueError('Root rotation or upper-body changed')
    foot=[model.body(s+'_ankle_pitch_link').id for s in ('left','right')];data=mujoco.MjData(model)
    old=np.arange(len(a))/fps;result={}
    for rate in (fps,50.):
        times=np.arange(int(np.floor(old[-1]*rate+1e-8))+1)/rate
        def poses(v):
            q=np.zeros((len(times),model.nq));q[:,:3]=np.array([np.interp(times,old,v[:,i]) for i in (1,2,3)]).T
            q[:,3:7]=Slerp(old,R.from_euler('xyz',v[:,4:7]))(times).as_quat()[:,[3,0,1,2]]
            q[:,7:]=np.array([np.interp(times,old,v[:,i]) for i in cols]).T
            pos=[];rot=[]
            for qi in q:
                data.qpos[:]=qi;mujoco.mj_kinematics(model,data)
                pos.append(data.xpos[foot].copy());rot.append(data.xmat[foot].reshape(2,3,3).copy())
            return np.array(pos),np.array(rot)
        pa,ra=poses(a);pb,rb=poses(b)
        angle=np.rad2deg((R.from_matrix(ra.reshape(-1,3,3)).inv()*R.from_matrix(rb.reshape(-1,3,3))).magnitude())
        result[str(int(rate))]=dict(max_position_drift_mm=float(np.linalg.norm(pb-pa,axis=-1).max()*1000),max_rotation_drift_deg=float(angle.max()))
    return dict(rates=result,timebase_exact=True,root_rotation_and_upper_body_exact=True,
        max_root_shift_m=float(np.linalg.norm(b[:,1:4]-a[:,1:4],axis=1).max()),
        passed=all(r['max_position_drift_mm']<=.1 and r['max_rotation_drift_deg']<=.01 for r in result.values()))


def audit_task(task):
    bundle,source,out,job=task;bundle=Path(bundle);source=Path(source);out=Path(out);api=api_for(bundle)
    from gear_sonic.scripts.audit_g1_style_retarget import evaluate
    from gear_sonic.scripts.h2_bvh_reference_worker import compare,ARM_METRICS,FOOT_METRICS
    import mujoco
    key=job['key'];old=read(source/'reports'/f'{key}.json');path=out/'refined/csv'/f'{key}.csv'
    if not api.raw_valid(Path(job['work']),job):raise ValueError('Raw BVH/G1 reference changed: '+key)
    if old['raw_files']!=job['raw_files']:raise ValueError('Baseline reference mismatch')
    baseline=source/'refined/csv'/f'{key}.csv'
    if digest(baseline)!=old['refined_sha256']:raise ValueError('Baseline changed')
    h={};g=old['audit_g1'];checks=[];physical=True
    for rate in (50,120):
        rr=str(rate);h[rr]=evaluate(path,Path(job['work'])/'h2'/f'{key}_trace.npz','h2',Path(job['reference']),rate)
        for side in ('Left','Right'):
            for section,metrics in [('arms',ARM_METRICS),('feet',FOOT_METRICS)]:
                checks.extend(dict(fps=rate,side=side,section=section,metric=k,stat=stat,h2=x,g1=y,passed=ok)
                    for k,stat,x,y,ok in compare(h[rr][section][side],g[rr][section][side],metrics))
        checks.extend(dict(fps=rate,section='torso',metric=k,stat=stat,h2=x,g1=y,passed=ok)
            for k,stat,x,y,ok in compare(h[rr],g[rr],('torso_lateral_error_deg','torso_orientation_error_deg')))
        c=h[rr]['native_checks'];physical &= c['joint_limit_excess_deg']<=1e-4 and c['max_urdf_velocity_ratio']<=1+1e-6 and c['shoulder_wrist_min_limit_margin_deg']>=2-1e-6
    model=mujoco.MjModel.from_xml_path(str(bundle/'code/gear_sonic/data/assets/robot_description/mjcf/h2.xml'))
    guard=trajectory_guard(model,baseline,path,120.)
    accepted=physical and guard['passed'] and all(c['passed'] for c in checks)
    status='accepted' if accepted else 'needs_review';dest=out/status/'csv'/f'{key}.csv';dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():
        if digest(dest)!=digest(path):raise ValueError('Admission output changed')
    else:os.link(path,dest)
    report=dict(key=key,status=status,output=str(dest),output_sha256=digest(dest),refined_sha256=digest(path),
        raw_files=job['raw_files'],checks=checks,physical_and_margin_pass=bool(physical),trajectory_guard=guard,
        audit_h2=h,audit_g1=g,source_bvh_sha256=job['bvh_sha256'],g1_retarget=False,
        g1_reference='reused baseline audit with verified raw files',training_started=False)
    save(out/'reports'/f'{key}.json',report);return key,status


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,default=Path('/home/arrc/lib/groot_wbc'))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--frame-chunk',type=int,default=131072)
    p.add_argument('--iterations',type=int,default=160)
    p.add_argument('--cpu-workers',type=int,default=16)
    p.add_argument('--limit',type=int,default=0,help='Smoke only: first N clips by length; zero selects all 48')
    a=p.parse_args();b=a.base;out=a.out.resolve()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1':raise ValueError('Use CUDA_VISIBLE_DEVICES=1; GPU 0 is reserved for training')
    if min(a.frame_chunk,a.iterations,a.cpu_workers)<1 or a.limit<0:raise ValueError('Invalid run settings')
    out.mkdir(parents=True,exist_ok=True);lock=(out/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    bundle=b/'artifacts/h2_bvh_gpu_staged_v5';source=b/'artifacts/h2_bvh_gpu_refine131k_116014_v6'
    pilot=b/'artifacts/h2_velocity_repair96_gpu1_v1';api=api_for(bundle);signature=api.verify_bundle()
    prev=read(pilot/'manifest.json');done=read(pilot/'complete.json')
    frames=read(bundle/'length_inventory.json')['frames']
    keys=sorted((k for k in prev['keys'] if k not in done['recovered_keys']),key=lambda k:(frames[k],k))
    if len(keys)!=48:raise ValueError('Expected the 48 prior velocity-only repair failures')
    if a.limit:keys=keys[:a.limit]
    controls=prev['controls'][:(2 if a.limit else 8)]
    helper=Path(__file__).with_name('h2_gpu_foot_preserving_retry.py')
    stamp=dict(source=str(source),source_complete_sha256=digest(source/'complete.json'),
        previous_pilot_manifest_sha256=digest(pilot/'manifest.json'),previous_pilot_complete_sha256=digest(pilot/'complete.json'),
        bundle_sha256=signature,code_sha256=digest(__file__),solver_sha256=digest(helper),
        keys=keys,controls=controls,iterations=a.iterations,frame_chunk=a.frame_chunk,gpu=1,
        foot_drift_limit_mm=.1,foot_rotation_drift_limit_deg=.01,root_translation_box_m=.1,g1_retarget=False)
    if (out/'manifest.json').exists() and read(out/'manifest.json')!=stamp:raise ValueError('Settings/provenance changed; use another output directory')
    save(out/'manifest.json',stamp)
    if (out/'complete.json').exists():print(json.dumps(read(out/'complete.json')));return
    spec=importlib.util.spec_from_file_location('h2_foot_retry_solver',helper);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation as R
    from gear_sonic.scripts.h2_gpu_reference_ik import GPUReferenceIK
    from gear_sonic.scripts.h2_bvh_reference_worker import read_csv,write_csv
    solver=GPUReferenceIK(bundle/'code/gear_sonic/data/assets/robot_description/mjcf/h2.xml')
    jobs={j['key']:j for j in api.jobs_for(source)}
    began=time.time();completed=0;group=[];count=0;(out/'refined/csv').mkdir(parents=True,exist_ok=True)
    def status(stage,**kwargs):
        value=dict(stage=stage,completed=completed,total=len(keys),seconds=round(time.time()-began,2),**kwargs)
        save(out/'progress.json',value);print(json.dumps(value),flush=True)
    def flush():
        nonlocal group,count,completed
        if not group:return
        starts=np.cumsum([0]+[len(g['q']) for g in group])[:-1].tolist()
        q=np.concatenate([g['q'] for g in group]);targets=np.concatenate([g['targets'] for g in group])
        answer=module.solve_lower(solver,q,targets,group[0]['names'],group[0]['fps'],starts,a.iterations,
            progress=lambda row:status('gpu_coupled_ik',batch_motions=len(group),batch_frames=count,**row))
        for g,start in zip(group,starts):
            raw=g['raw'].copy();v=answer[start:start+len(raw)];raw[:,1:4]=v[:,:3]
            for i,name in enumerate(solver.names):
                if any(k in name for k in ('hip','knee','ankle')):raw[:,g['cols'][i]]=v[:,7+i]
            key=g['key'];dest=out/'refined/csv'/f'{key}.csv'
            write_csv(dest.with_suffix('.partial'),g['header'],raw);dest.with_suffix('.partial').replace(dest)
            save(out/'refined/receipts'/f'{key}.json',dict(baseline_sha256=g['sha'],output_sha256=digest(dest),manifest_sha256=digest(out/'manifest.json')))
            completed+=1
        group=[];count=0;status('gpu_batch_complete')
    status('loading')
    for key in keys:
        job=jobs[key];baseline=source/'refined/csv'/f'{key}.csv';old=read(source/'reports'/f'{key}.json');sha=digest(baseline)
        if sha!=old['refined_sha256'] or not api.raw_valid(Path(job['work']),job):raise ValueError('Changed source: '+key)
        rp=out/'refined/receipts'/f'{key}.json';dest=out/'refined/csv'/f'{key}.csv'
        if rp.exists():
            r=read(rp)
            if r['baseline_sha256']!=sha or r['output_sha256']!=digest(dest) or r['manifest_sha256']!=digest(out/'manifest.json'):raise ValueError('Invalid resumed candidate')
            completed+=1;continue
        header,raw=read_csv(baseline);cols=[[n.removesuffix('_dof') for n in header].index(n) for n in solver.names]
        with np.load(Path(job['work'])/'h2'/f'{key}_trace.npz') as z:targets=z['targets'];names=z['names'].tolist();fps=float(z['fps'])
        if fps!=120. or len(raw)<2:raise ValueError('Expected audited 120Hz sequence')
        if group and (count+len(raw)>a.frame_chunk or names!=group[0]['names']):flush()
        q=np.c_[raw[:,1:4],R.from_euler('xyz',raw[:,4:7]).as_quat()[:,[3,0,1,2]],raw[:,cols]]
        group.append(dict(key=key,header=header,raw=raw,cols=cols,q=q,targets=targets,names=names,fps=fps,sha=sha));count+=len(raw)
    flush();torch.cuda.synchronize();status('auditing')
    for key in controls:shutil.copyfile(source/'accepted/csv'/f'{key}.csv',out/'refined/csv'/f'{key}.csv')
    pending=[]
    for key in keys+controls:
        rp=out/'reports'/f'{key}.json';dest=out/'refined/csv'/f'{key}.csv'
        if rp.exists():
            r=read(rp)
            if r['refined_sha256']!=digest(dest) or r['raw_files']!=jobs[key]['raw_files']:raise ValueError('Audit changed')
            continue
        pending.append((str(bundle),str(source),str(out),jobs[key]))
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.cpu_workers) as pool:
        for i,result in enumerate(pool.map(audit_task,pending),1):status('auditing',audit_completed=i,audit_pending=len(pending))
    recovered=[k for k in keys if read(out/'reports'/f'{k}.json')['status']=='accepted']
    regressed=[k for k in controls if read(out/'reports'/f'{k}.json')['status']!='accepted']
    result=dict(total=len(keys)+len(controls),retried=len(keys),recovered=len(recovered),recovered_keys=recovered,
        needs_review=len(keys)-len(recovered),unchanged_audit_controls=len(controls),control_regressions=regressed,
        seconds=round(time.time()-began,2),training_pool_unchanged=True,g1_retarget=False,
        optimization='GPU1',audit='CPU; native120/50Hz G1 comparison plus existing foot trajectory preservation',
        full_dataset_started=False,scope='Prior velocity-only repair failures; not all failed-motion categories')
    save(out/'complete.json',result);status('complete');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
