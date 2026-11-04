"""Independent raw-BVH/geometry audit for reconstructed H2 references.

Rebuilds goals from frozen trace + robot geometry, never from solver-exported
feasible targets. Old criteria and G1 benchmark failures remain explicit.
"""
from pathlib import Path
import argparse,csv,importlib.util,json,subprocess,sys,xml.etree.ElementTree as ET
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R,Slerp

BASE=Path('/home/arrc/lib/groot_wbc');KEY='dance_hiphop_kid_n_play_R_fast_001__A316_M'
ROOT=BASE/'artifacts/h2_foot_trajectory_v10';BUNDLE=BASE/'artifacts/h2_bvh_gpu_staged_v5'

def sha(p):
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def stats(v):
    v=np.asarray(v);return dict(mean=float(v.mean()),p95=float(np.percentile(v,95)),max=float(v.max()))

class Evaluation:
    def __init__(self,robot,csv_path,trace_path):
        name='h2.xml' if robot=='h2' else 'g1_29dof_rev_1_0.xml'
        self.robot=robot;self.m=mujoco.MjModel.from_xml_path(str(BUNDLE/'code/gear_sonic/data/assets/robot_description/mjcf'/name));self.d=mujoco.MjData(self.m)
        self.names=[self.m.joint(j).name for j in range(1,self.m.njnt)]
        with Path(csv_path).open() as f:rd=csv.reader(f);self.header=next(rd);self.raw=np.array(list(rd),float)
        cleaned=[n.removesuffix('_dof') for n in self.header];self.cols=[cleaned.index(n) for n in self.names]
        self.q=np.c_[self.raw[:,1:4],R.from_euler('xyz',self.raw[:,4:7]).as_quat()[:,[3,0,1,2]],self.raw[:,self.cols]]
        assert np.isfinite(self.q).all() and np.array_equal(self.raw[:,0],np.arange(len(self.raw)))
        z=np.load(trace_path);assert float(z['fps'])==120. and len(z['targets'])==len(self.q)
        self.targets=z['targets'][:,[z['names'].tolist().index(s+'Foot') for s in ('Left','Right')]].astype(float)
        self.old=np.arange(len(self.q))/120.
        self.bodies=[self.m.body(s+('_ankle_pitch_link' if robot=='h2' else '_ankle_roll_link')).id for s in ('left','right')]
        self.d.qpos[:]=self.q[0];mujoco.mj_kinematics(self.m,self.d);self.local=[];self.radii=[]
        for side,b in zip(('left','right'),self.bodies):
            if robot=='h2':
                g=self.m.geom(side+'_foot_collision').id
                signs=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
                pts=(signs*self.m.geom_size[g])@self.d.geom_xmat[g].reshape(3,3).T+self.d.geom_xpos[g];radii=np.zeros(8)
            else:
                ids=np.flatnonzero((self.m.geom_bodyid==b)&(self.m.geom_type==mujoco.mjtGeom.mjGEOM_SPHERE)&(self.m.geom_contype!=0));assert len(ids)==4
                pts=self.d.geom_xpos[ids];radii=self.m.geom_size[ids,0]
            self.local.append((pts-self.d.xpos[b])@self.d.xmat[b].reshape(3,3));self.radii.append(radii)

    def goal(self,times):
        p=np.stack([np.stack([np.interp(times,self.old,self.targets[:,s,j]) for j in range(3)],1) for s in range(2)],1)
        r=np.stack([Slerp(self.old,R.from_quat(self.targets[:,s,3:]))(times).as_matrix() for s in range(2)],1)
        h=np.stack([(np.einsum('nij,kj->nki',r[:,s],self.local[s])+p[:,s,None,:])[:,:,2].min(1) if self.robot=='h2' else ((np.einsum('nij,kj->nki',r[:,s],self.local[s])+p[:,s,None,:])[:,:,2]-self.radii[s]).min(1) for s in range(2)],1)
        lift=np.maximum(-h,0.);p[:,:,2]+=lift
        return p,r,lift

    def evaluate(self,rate):
        times=np.arange(int(np.floor(self.old[-1]*rate+1e-8))+1)/rate
        q=np.empty((len(times),self.m.nq));q[:,:3]=np.stack([np.interp(times,self.old,self.q[:,j]) for j in range(3)],1)
        q[:,3:7]=Slerp(self.old,R.from_quat(self.q[:,[4,5,6,3]]))(times).as_quat()[:,[3,0,1,2]]
        q[:,7:]=np.stack([np.interp(times,self.old,self.q[:,j]) for j in range(7,self.m.nq)],1)
        pp=[];rr=[];hh=[]
        for qi in q:
            self.d.qpos[:]=qi;mujoco.mj_kinematics(self.m,self.d)
            p=self.d.xpos[self.bodies].copy();r=self.d.xmat[self.bodies].reshape(2,3,3).copy();h=[]
            for s in range(2):h.append(float(((self.local[s]@r[s].T+p[s])[:,2]-self.radii[s]).min()))
            pp.append(p);rr.append(r);hh.append(h)
        pp,rr,hh=map(np.asarray,(pp,rr,hh));gp,gr,lift=self.goal(times)
        ep=np.linalg.norm(pp-gp,axis=-1)*1000;xy=np.linalg.norm(pp[:,:,:2]-gp[:,:,:2],axis=-1)*1000
        er=(R.from_matrix(gr.reshape(-1,3,3)).inv()*R.from_matrix(rr.reshape(-1,3,3))).magnitude().reshape(-1,2)*180/np.pi
        dyaw=np.arctan2(rr[:,:,1,0],rr[:,:,0,0])-np.arctan2(gr[:,:,1,0],gr[:,:,0,0]);yaw=abs(np.arctan2(np.sin(dyaw),np.cos(dyaw)))*180/np.pi
        grounded=lift>1e-8
        result=dict(samples=len(times),min_clearance_mm=float(hh.min()*1000),penetrating_samples=int((hh<-1e-7).sum()),grounded_target_samples=int(grounded.sum()),grounded_target_height_mm=stats(hh[grounded]*1000) if grounded.any() else None,grounded_target_above_0p5mm=int((hh[grounded]>.0005+1e-7).sum()),ground_lift_max_mm=float(lift.max()*1000),feet={s:dict(position_error_mm=stats(ep[:,i]),horizontal_error_mm=stats(xy[:,i]),rotation_error_deg=stats(er[:,i]),yaw_error_deg=stats(yaw[:,i])) for i,s in enumerate(('Left','Right'))})
        return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('candidate',type=Path);a=ap.parse_args();candidate=a.candidate;before=sha(candidate)
    # This separate evaluator regenerates the old source/G1 comparison report.
    subprocess.run([sys.executable,str(ROOT/'audit_candidates.py'),str(candidate)],check=True)
    legacy=json.loads((candidate.parent/'independent_audit.json').read_text())
    diagnosis=json.loads((ROOT/'source_foot_reference_diagnosis.json').read_text());trace=Path(diagnosis['trace']);work=Path(diagnosis['work'])
    h2=Evaluation('h2',candidate,trace);g1_csv=work/'g1/csv'/f'{KEY}.csv';g1_trace=work/'g1'/f'{KEY}_trace.npz';g1=Evaluation('g1',g1_csv,g1_trace)
    pins={str(p):sha(p) for p in [trace,g1_csv,g1_trace,BUNDLE/'code/gear_sonic/data/assets/robot_description/mjcf/h2.xml']}
    rates={};benchmark=[]
    for rate in (120,50,600):
        hh=h2.evaluate(rate);gg=g1.evaluate(rate);rates[str(rate)]=dict(h2=hh,g1_frozen_reference=gg)
        if rate!=600:
            for side in ('Left','Right'):
                for metric in ('position_error_mm','horizontal_error_mm','rotation_error_deg','yaw_error_deg'):
                    for stat in ('mean','p95','max'):
                        x=hh['feet'][side][metric][stat];y=gg['feet'][side][metric][stat]
                        benchmark.append(dict(rate=rate,side=side,metric=metric,stat=stat,h2=x,g1=y,passed=x<=y+1e-6))
    warm=Evaluation('h2',ROOT/'window_native_01'/f'{KEY}.csv',trace)
    upper=[i for i,n in enumerate(h2.names) if not any(s in n for s in ('hip','knee','ankle'))]
    upper_exact=np.array_equal(h2.raw[:,np.array(h2.cols)[upper]],warm.raw[:,np.array(warm.cols)[upper]])
    root_exact=np.array_equal(h2.raw[:,4:7],warm.raw[:,4:7]);ids_exact=np.array_equal(h2.raw[:,0],warm.raw[:,0])
    old_upper=[v for v in legacy['failed_g1_checks'] if v['section'] in ('arms','torso')]
    ground_pass=all(v['h2']['penetrating_samples']==0 for v in rates.values())
    contact_pass=all(v['h2']['grounded_target_above_0p5mm']==0 for v in rates.values())
    structural=bool(ids_exact and root_exact and upper_exact and legacy['physical_pass'] and legacy['root_metrics_pass'] and not old_upper and ground_pass)
    validated=structural and contact_pass
    failed=[v for v in benchmark if not v['passed']]
    assert before==sha(candidate) and all(sha(p)==v for p,v in pins.items())
    report=dict(reference_contract='h2_bvh_grounded_joint_feasible',key=KEY,candidate=str(candidate),candidate_sha256=before,source_pins=pins,goals_derived_from_raw_trace_and_geometry=True,solver_feasible_targets_used_as_quality_goal=False,floor_z_m=0.,clearance_epsilon_m=1e-7,grounded_target_height_tolerance_mm=.5,rates=rates,source_grounded_benchmark_checks=benchmark,source_grounded_benchmark_failed_checks=failed,source_grounded_benchmark_pass=not failed,frame_ids_exact=bool(ids_exact),upper_joint_values_exact_to_qualified_warm=bool(upper_exact),root_rotation_exact_to_qualified_warm=bool(root_exact),physical_pass=legacy['physical_pass'],root_metrics_pass=legacy['root_metrics_pass'],upper_g1_pass=not old_upper,ground_pass=ground_pass,grounded_target_height_pass=contact_pass,structural_reference_pass=structural,reconstructed_reference_validation_pass=validated,old_g1_pass=legacy['g1_pass'],old_legacy_admission_pass=legacy['legacy_admission_pass'],old_failed_g1_checks=legacy['failed_g1_checks'],old_reference_geometry=legacy['geometry'],scope='Kinematic reconstructed-reference validation only. Grounded source target is a geometry-derived proxy, not measured contact forces or policy success.',admission_performed=False,training_started=False,code_sha256=sha(__file__))
    save(candidate.parent/'bvh_reference_audit.json',report)
    print('NEW_REFERENCE_AUDIT',json.dumps({k:report[k] for k in ['candidate','physical_pass','ground_pass','grounded_target_height_pass','upper_g1_pass','root_metrics_pass','reconstructed_reference_validation_pass','source_grounded_benchmark_pass','old_legacy_admission_pass']}|dict(benchmark_failed_count=len(failed))),flush=True)

if __name__=='__main__':main()
