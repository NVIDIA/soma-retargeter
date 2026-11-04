"""Candidate-only audit: original G1 checks plus original foot-trajectory guard."""
from pathlib import Path
import os
import importlib.util

spec=importlib.util.spec_from_file_location('original_foot_audit',Path(__file__).with_name('run_h2_foot_preserving_reopt.py'))
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
read,save,digest,api_for=old.read,old.save,old.digest,old.api_for

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
    protected=[]  # Upper-body candidates are allowed; full G1/physical audit still applies.
    if not np.array_equal(a[:,[0,4,5,6]+protected],b[:,[0,4,5,6]+protected]):raise ValueError('Root rotation changed')
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
    return dict(rates=result,timebase_exact=True,root_rotation_exact=True,
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
    report=dict(recipe=job.get('recipe'),key=key,status=status,output=str(dest),output_sha256=digest(dest),refined_sha256=digest(path),
        raw_files=job['raw_files'],checks=checks,physical_and_margin_pass=bool(physical),trajectory_guard=guard,
        audit_h2=h,audit_g1=g,source_bvh_sha256=job['bvh_sha256'],g1_retarget=False,
        g1_reference='reused baseline audit with verified raw files',training_started=False)
    save(out/'reports'/f'{key}.json',report);return key,status

