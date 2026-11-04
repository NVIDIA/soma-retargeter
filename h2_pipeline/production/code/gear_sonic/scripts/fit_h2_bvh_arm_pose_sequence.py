"""Whole-clip kinematic arm IK: source directions, joint limits and continuity.

No inverse dynamics, force targets, PKL editing, or time warping.
"""
import argparse,csv,json,hashlib,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np,mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
from scipy.sparse import block_diag,kron,eye,diags,vstack
from scipy.ndimage import gaussian_filter1d
from gear_sonic.scripts.refine_h2_bvh_arm_targets import skew,arm_joint_bounds


def fit(model,q,side,source_directions,dt,initial,iterations=80,velocity_weight=200.,wrist_targets=None,wrist_weight=50.,smooth_sigma=4.,direction_soft_limit_deg=15.,excess_weight=0.,forearm_weight=2.,flex_weight=0.,flex_source_threshold_deg=180.,upper_weight=2.,upper_soft_limit_deg=None,forearm_soft_limit_deg=None,wrist_soft_limit_deg=180.,wrist_excess_weight=0.,joint_limit_margin_deg=2.):
    m=mujoco.MjModel.from_xml_path(str(model));d=mujoco.MjData(m);names=[m.joint(i).name for i in range(1,m.njnt)]
    js=np.array([i for i,n in enumerate(names) if n.startswith(side.lower()+'_') and any(s in n for s in ('shoulder','elbow','wrist'))]);dofs=6+js
    bodies=[m.body(side.lower()+'_'+s+'_link').id for s in ('shoulder_roll','elbow','wrist_yaw')];n=len(q);k=len(js)
    urdf=Path(model).resolve().parent.parent/'urdf/h2/h2.urdf'
    velocity={j.get('name'):float(j.find('limit').get('velocity')) for j in ET.parse(urdf).getroot().findall('joint') if j.find('limit') is not None}
    caps=np.tile([velocity[names[j]] for j in js],n-1)*dt*.98
    d1=kron(diags([-np.ones(n-1),np.ones(n-1)],[0,1],shape=(n-1,n)),eye(k),format='csr')
    d2=kron(diags([np.ones(n-2),-2*np.ones(n-2),np.ones(n-2)],[0,1,2],shape=(n-2,n)),eye(k),format='csr')
    jp=np.zeros((3,m.nv));jr=jp.copy();cache={}
    def geometry(x):
        if 'x' in cache and np.array_equal(cache['x'],x):return cache['r'],cache['j']
        values=x.reshape(n,k);errors=[];blocks=[]
        for i in range(n):
            d.qpos[:]=q[i];d.qpos[7+js]=values[i];mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
            v=np.diff(d.xpos[bodies],axis=0);lengths=np.linalg.norm(v,axis=1);u=v/lengths[:,None]
            wr=R.from_matrix(d.xmat[bodies[-1]].reshape(3,3));e=(wrist_targets[i].inv()*wr).as_rotvec()
            diff=u-source_directions[i];norm=np.linalg.norm(diff,axis=1);threshold=2*np.sin(np.deg2rad([direction_soft_limit_deg if upper_soft_limit_deg is None else upper_soft_limit_deg,direction_soft_limit_deg if forearm_soft_limit_deg is None else forearm_soft_limit_deg])/2)
            factor=np.maximum(1-threshold/np.maximum(norm,1e-12),0)
            flex=np.arccos(np.clip(u[0]@u[1],-1,1));human_flex=np.arccos(np.clip(source_directions[i,0]@source_directions[i,1],-1,1))
            flex_w=flex_weight if np.rad2deg(human_flex)<=flex_source_threshold_deg else 0.
            wrist_norm=np.linalg.norm(e);wrist_threshold=np.deg2rad(wrist_soft_limit_deg);wrist_factor=max(1-wrist_threshold/max(wrist_norm,1e-12),0)
            errors.append(np.r_[np.r_[diff[0]*upper_weight,diff[1]*forearm_weight],e*wrist_weight,(diff*factor[:,None]).ravel()*excess_weight,(flex-human_flex)*flex_w,e*wrist_factor*wrist_excess_weight]);jac=[]
            for body in bodies:mujoco.mj_jacBody(m,d,jp,jr,body);jac.append(jp[:,dofs].copy())
            angle=np.linalg.norm(e);kk=skew(e);aa=1/12 if angle<1e-4 else (1-.5*angle/np.tan(.5*angle))/angle**2
            orientation_jac=(np.eye(3)-.5*kk+aa*kk@kk)@wrist_targets[i].as_matrix().T@jr[:,dofs]*wrist_weight
            direction_jac=[(np.eye(3)-np.outer(u[j],u[j]))/lengths[j]@(jac[j+1]-jac[j]) for j in range(2)]
            excess_jac=[]
            for j in range(2):
                b=np.zeros((3,3)) if norm[j]<=threshold[j] else np.eye(3)*(1-threshold[j]/norm[j])+threshold[j]*np.outer(diff[j],diff[j])/norm[j]**3
                excess_jac.append(b@direction_jac[j]*excess_weight)
            flex_jac=-(u[1]@direction_jac[0]+u[0]@direction_jac[1])/max(np.sin(flex),1e-7)*flex_w
            wrist_b=np.zeros((3,3)) if wrist_norm<=wrist_threshold else np.eye(3)*(1-wrist_threshold/wrist_norm)+wrist_threshold*np.outer(e,e)/wrist_norm**3
            wrist_peak_jac=wrist_b@(orientation_jac/wrist_weight)*wrist_excess_weight
            blocks.append(np.vstack([direction_jac[0]*upper_weight,direction_jac[1]*forearm_weight,orientation_jac,*excess_jac,flex_jac[None,:],wrist_peak_jac]))
        rr=np.concatenate(errors);jj=block_diag(blocks,format='csr');cache.update(x=x.copy(),r=rr,j=jj);return rr,jj
    anchor=initial[:,7+js].ravel();identity=eye(n*k,format='csr')
    def residual(x):
        geo,_=geometry(x);diff=d1@x;over=np.maximum(abs(diff)-caps,0)*np.sign(diff)
        return np.r_[geo,(x-anchor)*.003,d2@x*.2,over*velocity_weight]
    def jac(x):
        _,g=geometry(x);active=(abs(d1@x)>caps).astype(float)
        return vstack([g,identity*.003,d2*.2,diags(active)@d1*velocity_weight],format='csr')
    joint_lo,joint_hi=arm_joint_bounds(m,1+js,joint_limit_margin_deg)
    lo=np.tile(joint_lo,n);hi=np.tile(joint_hi,n)
    x0=(gaussian_filter1d(initial[:,7+js],smooth_sigma,axis=0) if smooth_sigma>0 else initial[:,7+js]).ravel();x0=np.clip(x0,lo+1e-9,hi-1e-9)
    initial_cost=float(np.linalg.norm(residual(x0)))
    opt=least_squares(residual,x0,jac=jac,bounds=(lo,hi),max_nfev=iterations,ftol=1e-8,xtol=1e-10,gtol=1e-8,x_scale='jac',tr_solver='lsmr',tr_options={'maxiter':1200,'atol':1e-8,'btol':1e-8},verbose=0)
    result=q.copy();result[:,7+js]=opt.x.reshape(n,k)
    diagnostics=dict(side=side,success=bool(opt.success),optimality=float(opt.optimality),status=int(opt.status),iterations=int(opt.nfev),initial_cost=initial_cost,final_cost=float(np.linalg.norm(residual(opt.x))),
                     max_direction_residual=float(np.linalg.norm(geometry(opt.x)[0].reshape(n,19)[:,:6]/np.array([upper_weight]*3+[forearm_weight]*3),axis=1).max()),max_velocity_rad_s=float(abs(d1@opt.x).max()/dt),max_velocity_limit_ratio=float(np.max(abs(d1@opt.x)/caps)),max_wrist_error_deg=float(np.rad2deg(np.linalg.norm(geometry(opt.x)[0].reshape(n,19)[:,6:9]/wrist_weight,axis=1).max())))
    return result,diagnostics


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--trace',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--iterations',type=int,default=80);p.add_argument('--velocity-weight',type=float,default=200.);p.add_argument('--wrist-weight',type=float,default=50.);p.add_argument('--smooth-sigma',type=float,default=4.);p.add_argument('--sides',nargs='+',choices=['Left','Right'],default=['Left','Right']);p.add_argument('--direction-soft-limit-deg',type=float,default=15.);p.add_argument('--excess-weight',type=float,default=0.);p.add_argument('--forearm-weight',type=float,default=2.);p.add_argument('--flex-weight',type=float,default=0.);p.add_argument('--flex-source-threshold-deg',type=float,default=180.);p.add_argument('--upper-weight',type=float,default=2.);p.add_argument('--upper-soft-limit-deg',type=float);p.add_argument('--forearm-soft-limit-deg',type=float);p.add_argument('--wrist-soft-limit-deg',type=float,default=180.);p.add_argument('--wrist-excess-weight',type=float,default=0.);p.add_argument('--joint-limit-margin-deg',type=float,default=2.);a=p.parse_args()
    script_source=Path(__file__).read_bytes()
    a.out.mkdir(parents=True,exist_ok=False);(a.out/'csv').mkdir()
    with a.input.open() as f:rd=csv.reader(f);header=next(rd);raw=np.array(list(rd),float)
    m=mujoco.MjModel.from_xml_path(str(a.model));names=[m.joint(i).name for i in range(1,m.njnt)];clean=[x.removesuffix('_dof') for x in header];cols=[clean.index(n) for n in names]
    q=np.c_[raw[:,1:4],R.from_euler('xyz',raw[:,4:7]).as_quat()[:,[3,0,1,2]],raw[:,cols]];initial=q.copy()
    t=np.load(a.trace);source=t['source_world'];sn=t['source_names'].tolist();dt=1/float(t['fps']);rows=[]
    if len(q)!=len(source):raise ValueError('Frame mismatch')
    for side in a.sides:
        hp=source[:,[sn.index(side+s) for s in ('Arm','ForeArm','Hand')],:3];v=np.diff(hp,axis=1);v/=np.linalg.norm(v,axis=-1)[...,None]
        wr=R.from_quat(t['targets'][:,t['names'].tolist().index(side+'Hand'),3:])
        q,r=fit(a.model,q,side,v,dt,initial,a.iterations,a.velocity_weight,wr,a.wrist_weight,a.smooth_sigma,a.direction_soft_limit_deg,a.excess_weight,a.forearm_weight,a.flex_weight,a.flex_source_threshold_deg,a.upper_weight,a.upper_soft_limit_deg,a.forearm_soft_limit_deg,a.wrist_soft_limit_deg,a.wrist_excess_weight,a.joint_limit_margin_deg);rows.append(r);print(json.dumps(r),flush=True)
    output=raw.copy();output[:,cols]=q[:,7:]
    keep=[i for i,x in enumerate(clean) if not any(n in x for n in ('shoulder','elbow','wrist'))]
    assert np.array_equal(output[:,keep],raw[:,keep])
    with (a.out/'csv'/a.input.name).open('w') as f:w=csv.writer(f);w.writerow(header);w.writerows(output)
    (a.out/'report.json').write_text(json.dumps(dict(source=str(a.input),source_sha256=hashlib.sha256(a.input.read_bytes()).hexdigest(),results=rows,joint_limit_margin_deg=a.joint_limit_margin_deg,upper_soft_limit_deg=a.upper_soft_limit_deg,forearm_soft_limit_deg=a.forearm_soft_limit_deg,wrist_soft_limit_deg=a.wrist_soft_limit_deg,wrist_excess_weight=a.wrist_excess_weight,upper_weight=a.upper_weight,flex_source_threshold_deg=a.flex_source_threshold_deg,flex_weight=a.flex_weight,forearm_weight=a.forearm_weight,excess_weight=a.excess_weight,direction_soft_limit_deg=a.direction_soft_limit_deg,wrist_weight=a.wrist_weight,smooth_sigma=a.smooth_sigma,velocity_margin=.98,velocity_weight=a.velocity_weight,scope=__doc__,non_arm_columns_preserved=True,published=False),indent=2)+'\n')
    (a.out/'runner.py').write_bytes(script_source)
if __name__=='__main__':main()
