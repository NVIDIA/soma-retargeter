"""Isolated native foot-pose projection; old admission is measured separately.

No original CSV, training catalog, loader, or admission tolerance is changed.
Only root translation and leg joints inside diagnosed windows are optimized.
"""
from pathlib import Path
import argparse, csv, hashlib, importlib.util, json, os, sys, time
import xml.etree.ElementTree as ET
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R, Slerp

KEY = 'dance_hiphop_kid_n_play_R_fast_001__A316_M'
BASE = Path('/home/arrc/lib/groot_wbc')

def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def save(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

def skew(v):
    x,y,z=v
    return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])

def log_jac(v):
    a=np.linalg.norm(v); k=skew(v)
    c=1/12 if a<1e-4 else (1-.5*a/np.tan(.5*a))/a**2
    return np.eye(3)-.5*k+c*k@k

class FootModel:
    def __init__(self, path):
        import mujoco
        self.mj=mujoco; self.m=mujoco.MjModel.from_xml_path(str(path))
        self.d=mujoco.MjData(self.m)
        self.names=[self.m.joint(i).name for i in range(1,self.m.njnt)]
        self.legs=np.array([j for j,n in enumerate(self.names) if any(k in n for k in ('hip','knee','ankle'))])
        self.coords=np.r_[np.arange(3),7+self.legs]
        self.dofs=np.r_[np.arange(3),6+self.legs]
        self.bodies=[self.m.body(s+'_ankle_pitch_link').id for s in ('left','right')]
        self.geom=[self.m.geom(s+'_foot_collision').id for s in ('left','right')]
        self.corner_signs=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])

    def poses(self, q):
        ps=[]; rs=[]; heights=[]
        for qi in q:
            self.d.qpos[:]=qi; self.mj.mj_kinematics(self.m,self.d)
            ps.append(self.d.xpos[self.bodies].copy())
            rs.append(self.d.xmat[self.bodies].reshape(2,3,3).copy())
            heights.append([float(((self.corner_signs*self.m.geom_size[g])@self.d.geom_xmat[g].reshape(3,3).T+self.d.geom_xpos[g])[:,2].min()) for g in self.geom])
        return np.array(ps),np.array(rs),np.array(heights)

    def error_jac(self, q, target_p, target_r):
        self.d.qpos[:]=q; self.mj.mj_kinematics(self.m,self.d); self.mj.mj_comPos(self.m,self.d)
        residual=[]; blocks=[]
        for i,b in enumerate(self.bodies):
            e=(R.from_matrix(target_r[i]).inv()*R.from_matrix(self.d.xmat[b].reshape(3,3))).as_rotvec()
            jp=np.zeros((3,self.m.nv)); jr=jp.copy()
            self.mj.mj_jacBody(self.m,self.d,jp,jr,b)
            residual.extend([self.d.xpos[b]-target_p[i],e])
            blocks.extend([jp[:,self.dofs],log_jac(e)@target_r[i].T@jr[:,self.dofs]])
        return np.concatenate(residual),np.vstack(blocks)

    def project(self, q, target_p, target_r, anchor):
        q=q.copy(); x0=q[self.coords].copy()
        low=np.r_[anchor[:3]-.1,self.m.jnt_range[1+self.legs,0]+1e-6]
        high=np.r_[anchor[:3]+.1,self.m.jnt_range[1+self.legs,1]-1e-6]
        scale=np.tile([2000.]*3+[1/np.deg2rad(.1)]*3,2)
        prior=np.r_[[.005]*3,[.00005]*12]
        def fun(x, jac=False):
            v=q.copy(); v[self.coords]=x
            e,j=self.error_jac(v,target_p,target_r)
            return np.vstack([j*scale[:,None],np.diag(prior)]) if jac else np.r_[e*scale,(x-x0)*prior]
        opt=least_squares(fun,np.clip(x0,low+1e-10,high-1e-10),jac=lambda x:fun(x,True),bounds=(low,high),max_nfev=100,ftol=1e-11,xtol=1e-11,gtol=1e-10)
        q[self.coords]=opt.x
        return q,dict(nfev=opt.nfev,success=bool(opt.success),cost=float(opt.cost))

def load(path, model):
    with Path(path).open() as f:
        rd=csv.reader(f); h=next(rd); raw=np.array(list(rd),float)
    clean=[n.removesuffix('_dof') for n in h];cols=[clean.index(n) for n in model.names]
    q=np.c_[raw[:,1:4],R.from_euler('xyz',raw[:,4:7]).as_quat()[:,[3,0,1,2]],raw[:,cols]]
    assert np.array_equal(raw[:,0],np.arange(len(raw))) and np.isfinite(q).all()
    return h,raw,cols,q

def sample(q, times, fps=120.):
    old=np.arange(len(q))/fps
    result=np.empty((len(times),q.shape[1]))
    result[:,:3]=np.stack([np.interp(times,old,q[:,i]) for i in range(3)],1)
    result[:,3:7]=Slerp(old,R.from_quat(q[:,[4,5,6,3]]))(times).as_quat()[:,[3,0,1,2]]
    result[:,7:]=np.stack([np.interp(times,old,q[:,i]) for i in range(7,q.shape[1])],1)
    return result

def compare_geometry(model, original, candidate):
    native_p,native_r,_=model.poses(original);old=np.arange(len(original))/120.;result={}
    for rate in (120,50):
        times=np.arange(int(np.floor(old[-1]*rate+1e-8))+1)/rate
        p,r,h=model.poses(sample(candidate,times));lp,lr,lh=model.poses(sample(original,times))
        np_=np.stack([np.stack([np.interp(times,old,native_p[:,s,i]) for i in range(3)],1) for s in range(2)],1)
        nr=np.stack([Slerp(old,R.from_matrix(native_r[:,s]))(times).as_matrix() for s in range(2)],1)
        def delta(gp,gr):
            ep=np.linalg.norm(p-gp,axis=-1)*1000
            er=(R.from_matrix(gr.reshape(-1,3,3)).inv()*R.from_matrix(r.reshape(-1,3,3))).magnitude().reshape(-1,2)*180/np.pi
            return dict(position_mm=float(ep.max()),rotation_deg=float(er.max()),position_peak_sample=list(map(int,np.unravel_index(ep.argmax(),ep.shape))))
        result[str(rate)]=dict(legacy=delta(lp,lr),native_pose=delta(np_,nr),legacy_zero_clearance_crossings=int(((h<=0)!=(lh<=0)).sum()))
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    out=a.out;out.mkdir(parents=True,exist_ok=False)
    bundle=BASE/'artifacts/h2_bvh_gpu_staged_v5';model=FootModel(bundle/'code/gear_sonic/data/assets/robot_description/mjcf/h2.xml')
    source=BASE/'artifacts/h2_bvh_gpu_refine131k_116014_v6/refined/csv'/f'{KEY}.csv'
    warm=BASE/'artifacts/h2_compound_joint_v9/root_seeded_local_bvh/refined/csv'/f'{KEY}.csv'
    h,raw,cols,original=load(source,model);hh,wr,_,candidate=load(warm,model);assert hh==h and raw.shape==wr.shape
    info=json.loads((BASE/'artifacts/h2_compound_joint_v9/root_candidate_windows.json').read_text())
    mask=np.zeros(len(raw),bool)
    for lo,hi in info['failure_windows_native_frames']:mask[lo:hi+1]=True
    # The older all-clip optimizer modified healthy lower-body frames. Freeze them exactly.
    candidate[np.ix_(~mask,model.coords)]=original[np.ix_(~mask,model.coords)]
    target_p,target_r,_=model.poses(original);begin=time.time();iters=[]
    # Verify the analytic Jacobian independently before running the actual optimization.
    frame=int(np.flatnonzero(mask)[len(np.flatnonzero(mask))//2]);q=candidate[frame].copy();e,j=model.error_jac(q,target_p[frame],target_r[frame]);num=np.empty_like(j);eps=1e-6
    for k,idx in enumerate(model.coords):
        plus=q.copy();minus=q.copy();plus[idx]+=eps;minus[idx]-=eps
        num[:,k]=(model.error_jac(plus,target_p[frame],target_r[frame])[0]-model.error_jac(minus,target_p[frame],target_r[frame])[0])/(2*eps)
    error=float(abs(j-num).max());assert error<1e-5,error
    for i in np.flatnonzero(mask):
        candidate[i],meta=model.project(candidate[i],target_p[i],target_r[i],original[i]);iters.append(meta)
    result=wr.copy();result[:,1:4]=candidate[:,:3];result[:,cols]=candidate[:,7:]
    path=out/f'{KEY}.csv'
    with path.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(h);w.writerows(result)
    speeds={j.get('name'):float(j.find('limit').get('velocity')) for j in ET.parse(bundle/'code/gear_sonic/data/assets/robot_description/urdf/h2/h2.urdf').getroot().findall('joint') if j.find('limit') is not None}
    ratios=abs(np.diff(candidate[:,7:],axis=0))*120/np.array([speeds[n] for n in model.names])
    worst=np.unravel_index(ratios.argmax(),ratios.shape)
    report=dict(key=KEY,source_csv=str(source),source_sha256=digest(source),warm_csv=str(warm),warm_sha256=digest(warm),candidate_csv=str(path),candidate_sha256=digest(path),code_sha256=digest(__file__),windows=info['failure_windows_native_frames'],jacobian_fd_max_error=error,seconds=time.time()-begin,projected_frames=int(mask.sum()),geometry=compare_geometry(model,original,candidate),native_max_speed_ratio=float(ratios.max()),native_speed_worst=dict(frame=int(worst[0]),joint=model.names[worst[1]]),healthy_lower_exact=bool(np.array_equal(candidate[np.ix_(~mask,model.coords)],original[np.ix_(~mask,model.coords)])),root_rotation_and_upper_unchanged_from_warm=bool(np.array_equal(result[:,[0,4,5,6]+[cols[i] for i in range(31) if i not in model.legs]],wr[:,[0,4,5,6]+[cols[i] for i in range(31) if i not in model.legs]])),admission_performed=False,training_started=False)
    save(out/'projection_report.json',report);save(out/'iterations.json',iters)
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
