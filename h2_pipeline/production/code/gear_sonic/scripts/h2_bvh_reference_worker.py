"""CPU refinement and independent H2/G1 admission for fresh batched BVH outputs."""
import csv,hashlib,json,shutil,time,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np,mujoco
from scipy.spatial.transform import Rotation as R
from gear_sonic.scripts.audit_g1_style_retarget import evaluate
from gear_sonic.scripts.h2_foot_pose_ik import FootPoseSolver
from gear_sonic.scripts.refine_h2_bvh_arm_targets import ArmSolver
from gear_sonic.scripts.fit_h2_bvh_arm_pose_sequence import fit
from gear_sonic.scripts.refine_h2_wrist_interpolation import refine
from gear_sonic.scripts.select_h2_bvh_pose_candidates import ARM_METRICS,FOOT_METRICS
ROOT=Path(__file__).resolve().parents[2]


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read_csv(path):
    with Path(path).open() as f:r=csv.reader(f);h=next(r);x=np.array(list(r),float)
    return h,x

def write_csv(path,header,raw):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w') as f:w=csv.writer(f);w.writerow(header);w.writerows(raw)

def compare(h,g,metrics):
    values=[]
    for k in metrics:
        if h[k] is None:
            if g[k] is not None:raise ValueError('Source mask mismatch')
            continue
        for stat in ('mean','p95','max'):
            x,y=h[k][stat],g[k][stat];values.append((k,stat,x,y,x<=y+1e-6))
    return values

def rank(checks):
    return (sum(not ok for _,stat,x,y,ok in checks if stat!='max'),sum(not ok for _,stat,x,y,ok in checks if stat=='max'),max([x/(y+1e-6) for _,stat,x,y,ok in checks]+[0]))


def process_one(job):
    started=time.time();key=job['key'];work=Path(job['work']);out=Path(job['out']);reference=Path(job['reference']);model=ROOT/'gear_sonic/data/assets/robot_description/mjcf/h2.xml'
    try:
        header,raw=read_csv(work/'h2/csv'/f'{key}.csv');trace=work/'h2'/f'{key}_trace.npz';t=np.load(trace);tn=t['names'].tolist();sn=t['source_names'].tolist();fps=float(t['fps'])
        m=mujoco.MjModel.from_xml_path(str(model));names=[m.joint(j).name for j in range(1,m.njnt)];clean=[x.removesuffix('_dof') for x in header];cols=[clean.index(n) for n in names]
        caps_by_name={j.get('name'):float(j.find('limit').get('velocity')) for j in ET.parse(ROOT/'gear_sonic/data/assets/robot_description/urdf/h2/h2.urdf').getroot().findall('joint') if j.find('limit') is not None};lower_js=[i for i,n in enumerate(names) if any(k in n for k in ('hip','knee','ankle','waist'))];lower_caps=np.array([caps_by_name[names[j]] for j in lower_js])
        def lower_ok(q):return len(q)<2 or float((abs(np.diff(q[:,7+np.array(lower_js)],axis=0))*fps/lower_caps).max())<=1+1e-6
        original=np.c_[raw[:,1:4],R.from_euler('xyz',raw[:,4:7]).as_quat()[:,[3,0,1,2]],raw[:,cols]];temp=work/'post'/key;temp.mkdir(parents=True,exist_ok=True);candidate=temp/(key+'.csv')
        if len(raw)!=len(t['targets']):raise ValueError('Frame count mismatch')
        g={rate:evaluate(work/'g1/csv'/f'{key}.csv',work/'g1'/f'{key}_trace.npz','g1',reference,rate) for rate in (50,120)}
        def audit(q):
            data=raw.copy();data[:,1:4]=q[:,:3];data[:,4:7]=R.from_quat(q[:,[4,5,6,3]]).as_euler('xyz');data[:,cols]=q[:,7:];write_csv(candidate,header,data)
            return {rate:evaluate(candidate,trace,'h2',reference,rate) for rate in (50,120)}
        def arm_checks(a,side):return [v for rate in (50,120) for v in compare(a[rate]['arms'][side],g[rate]['arms'][side],ARM_METRICS)]
        def foot_checks(a):return [v for rate in (50,120) for side in ('Left','Right') for v in compare(a[rate]['feet'][side],g[rate]['feet'][side],FOOT_METRICS)]
        foot=FootPoseSolver(model);base_audit=audit(original);foot_options=[]
        for reverse in (False,True):
            q=original.copy();previous=None;full=None
            for i in (range(len(q)-1,-1,-1) if reverse else range(len(q))):
                goal=t['targets'][i,[tn.index('LeftFoot'),tn.index('RightFoot')]];result,previous,diag=foot.solve(original[i],goal,previous)
                if max(diag['position_error_mm'])>1 or max(diag['rotation_error_deg'])>.1:result,diag=foot.solve_with_pelvis_rotation(original[i],goal,R.from_quat(t['targets'][i,tn.index('Chest'),3:]),full)
                full=result.copy();q[i]=result
            a=audit(q);foot_options.append(((not lower_ok(q),)+rank(foot_checks(a)),q,a,reverse))
            if foot_options[-1][0][0:3]==(False,0,0):break
        _,q,a,reverse=min(foot_options,key=lambda x:x[0]);stages=[dict(stage='feet',reverse=reverse)]
        canonical=np.load(reference)['qpos'];solver=ArmSolver(model,velocity_cap=28,geometry_weight=2,wrist_weight=50,elbow_weight=0,continuity_weight=.03,canonical_weight=0,multistart=3,joint_limit_margin_deg=2)
        for side in ('Left','Right'):
            js=np.array([j for j,n in enumerate(names) if n.startswith(side.lower()+'_') and any(k in n for k in ('shoulder','elbow','wrist'))]);hp=t['source_world'][:,[sn.index(side+n) for n in ('Arm','ForeArm','Hand')],:3];directions=np.diff(hp,axis=1);directions/=np.linalg.norm(directions,axis=-1)[...,None];wr=R.from_quat(t['targets'][:,tn.index(side+'Hand'),3:]);options=[]
            for backwards in (False,True):
                trial=q.copy();prev=None
                for i in (range(len(q)-1,-1,-1) if backwards else range(len(q))):trial[i],prev,_=solver.solve(q[i],side,directions[i],wr[i],prev,canonical[7+js],1/fps)
                ta=audit(trial);options.append((rank(arm_checks(ta,side)),trial,ta,'reverse' if backwards else 'forward'))
                if options[-1][0][0:2]==(0,0):break
            score,best,ba,stage=min(options,key=lambda x:x[0])
            if score[0:2]!=(0,0) and len(q)>2:
                trial,diag=fit(model,best,side,directions,1/fps,best,iterations=180,velocity_weight=1500,wrist_targets=wr,wrist_weight=20,smooth_sigma=0,forearm_weight=8,flex_weight=6,flex_source_threshold_deg=15,upper_weight=3,joint_limit_margin_deg=2)
                ta=audit(trial);options.append((rank(arm_checks(ta,side)),trial,ta,'sequence'))
                score,best,ba,stage=min(options,key=lambda x:x[0])
            if ba[120]['arms'][side]['wrist_orientation_error_deg']['max']<=g[120]['arms'][side]['wrist_orientation_error_deg']['max']+1e-6 and ba[50]['arms'][side]['wrist_orientation_error_deg']['max']>g[50]['arms'][side]['wrist_orientation_error_deg']['max']+1e-6 and len(q)>1:
                cap=max(1e-5,.95*min(g[50]['arms'][side]['wrist_orientation_error_deg']['max'],g[120]['arms'][side]['wrist_orientation_error_deg']['max']))
                trial,_=refine(model,best,wr,side,fps,50,cap,2)
                ta=audit(trial);options.append((rank(arm_checks(ta,side)),trial,ta,'resample'))
                score,best,ba,stage=min(options,key=lambda x:x[0])
            q=best;a=ba;stages.append(dict(stage='arm',side=side,method=stage,score=score))
        a=audit(q);checks=[];physical=True
        for rate in (50,120):
            for side in ('Left','Right'):
                for section,metrics in [('arms',ARM_METRICS),('feet',FOOT_METRICS)]:
                    checks.extend(dict(fps=rate,side=side,section=section,metric=k,stat=stat,h2=x,g1=y,passed=ok) for k,stat,x,y,ok in compare(a[rate][section][side],g[rate][section][side],metrics))
            checks.extend(dict(fps=rate,section='torso',metric=k,stat=stat,h2=x,g1=y,passed=ok) for k,stat,x,y,ok in compare(a[rate],g[rate],('torso_lateral_error_deg','torso_orientation_error_deg')))
            c=a[rate]['native_checks'];physical &= c['joint_limit_excess_deg']<=1e-4 and c['max_urdf_velocity_ratio']<=1+1e-6 and c['shoulder_wrist_min_limit_margin_deg']>=2-1e-6
        status='accepted' if physical and all(c['passed'] for c in checks) else 'needs_review';dest=out/status/'csv'/f'{key}.csv';dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(candidate,dest.with_suffix('.partial'));dest.with_suffix('.partial').replace(dest)
        report=dict(key=key,status=status,source_bvh_sha256=job['bvh_sha256'],output=str(dest),output_sha256=digest(dest),physical_and_margin_pass=bool(physical),checks=checks,audit_h2=a,audit_g1=g,stages=stages,seconds=time.time()-started,source='fresh BVH',g1_reference=dict(source=job.get('g1_csv'),sha256=job.get('g1_csv_sha256'),retargeted=False),training_started=False)
    except Exception as exc:
        import traceback
        report=dict(key=key,status='error',error=str(exc),traceback=traceback.format_exc(),seconds=time.time()-started)
    report_dir=out/'reports';report_dir.mkdir(exist_ok=True,parents=True);tmp=report_dir/(key+'.partial');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(report_dir/(key+'.json'))
    return dict(key=key,status=report['status'],seconds=round(report['seconds'],2))
