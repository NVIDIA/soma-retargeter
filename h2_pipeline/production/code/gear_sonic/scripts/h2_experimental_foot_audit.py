"""Explicit 0.5 mm / 0.1 degree experimental foot guard; original G1 gates intact."""
from pathlib import Path
import importlib.util
spec=importlib.util.spec_from_file_location('experimental_base_audit',Path(__file__).with_name('h2_expansion_audit.py'));base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
original_guard=base.trajectory_guard
POSITION_MM=.5
ROTATION_DEG=.1

def within_guard(rates):
 return bool(rates) and all(v['max_position_drift_mm']<=POSITION_MM and v['max_rotation_drift_deg']<=ROTATION_DEG for v in rates.values())

def clearance_diagnostics(model,baseline,candidate,fps):
 """FK-only support/swing diagnostics; no contact-force or dynamics claim."""
 import csv,itertools
 import numpy as np
 import mujoco
 from scipy.spatial.transform import Rotation as R,Slerp
 def load(path):
  with Path(path).open() as f:
   reader=csv.reader(f);header=[s.removesuffix('_dof') for s in next(reader)];return header,np.array(list(reader),float)
 header,a=load(baseline);hh,b=load(candidate);assert header==hh and a.shape==b.shape
 cols=[header.index(model.joint(i).name) for i in range(1,model.njnt)];geoms=[model.geom(side+'_foot_collision').id for side in ('left','right')]
 corners=[np.array(list(itertools.product((-1,1),repeat=3)))*model.geom_size[g] for g in geoms];data=mujoco.MjData(model);old=np.arange(len(a))/fps;summary={}
 for rate in (120.,50.):
  times=np.arange(int(np.floor(old[-1]*rate+1e-8))+1)/rate
  def feet(v):
   q=np.zeros((len(times),model.nq));q[:,:3]=np.array([np.interp(times,old,v[:,i]) for i in (1,2,3)]).T;q[:,3:7]=Slerp(old,R.from_euler('xyz',v[:,4:7]))(times).as_quat()[:,[3,0,1,2]];q[:,7:]=np.array([np.interp(times,old,v[:,i]) for i in cols]).T;out=[]
   for qi in q:
    data.qpos[:]=qi;mujoco.mj_kinematics(model,data);out.append(np.array([local@data.geom_xmat[g].reshape(3,3).T+data.geom_xpos[g] for g,local in zip(geoms,corners)]))
   return np.array(out)
  pa,pb=feet(a),feet(b);ha,hb=pa[:,:,:,2].min(2),pb[:,:,:,2].min(2);delta=pb-pa;centers_a=pa.mean(2);centers_b=pb.mean(2);speed=np.linalg.norm(np.diff(centers_a,axis=0),axis=-1)*rate;speed=np.vstack([speed[:1],speed]);support=(ha<=.01)&(speed<=.1);swing=~support
  def peak(mask):return float(np.linalg.norm(delta,axis=-1).max(2)[mask].max()*1000) if mask.any() else None
  summary[str(int(rate))]=dict(max_sole_corner_drift_mm=float(np.linalg.norm(delta,axis=-1).max()*1000),max_clearance_change_mm=float(abs(hb-ha).max()*1000),max_added_penetration_mm=float(np.maximum(np.minimum(ha,0)-np.minimum(hb,0),0).max()*1000),support_proxy_frames=int(support.sum()),swing_proxy_frames=int(swing.sum()),max_support_corner_drift_mm=peak(support),max_swing_corner_drift_mm=peak(swing),zero_clearance_crossing_frames=int(((ha<=0)!=(hb<=0)).sum()))
 return dict(rates=summary,contact_labels_available=False,scope='Kinematic proxy from baseline sole height <=10mm and foot speed <=0.1m/s; original timebase fixed. Does not prove actual contact forces or slip.')

def trajectory_guard(model,baseline,candidate,fps):
 result=original_guard(model,baseline,candidate,fps);result['strict_original_passed']=result['passed'];result['policy']='experimental_0p5mm_0p1deg_v1';result['limits']=dict(max_position_drift_mm=POSITION_MM,max_rotation_drift_deg=ROTATION_DEG);result['passed']=within_guard(result['rates'])
 if result['passed']:result['clearance_diagnostics']=clearance_diagnostics(model,baseline,candidate,fps)
 return result

base.trajectory_guard=trajectory_guard

def audit_task(task):
 key,status=base.audit_task(task);out=Path(task[2]);path=out/'reports'/f'{key}.json';r=base.read(path);r.update(foot_policy='experimental_0p5mm_0p1deg_v1',g1_criteria_unchanged=True,training_started=False);base.save(path,r);return key,status
