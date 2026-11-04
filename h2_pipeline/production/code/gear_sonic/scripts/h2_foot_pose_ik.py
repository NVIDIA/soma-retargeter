"""H2 kinematic foot-pose IK with a translational pelvis correction.

Foot transforms, root/torso rotations and timestamps are input constraints.
No dynamics, contact relabeling, time warping, or learned-policy changes.
"""
import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
from gear_sonic.scripts.refine_h2_bvh_arm_targets import skew


class FootPoseSolver:
    def __init__(self,model):
        self.m=mujoco.MjModel.from_xml_path(str(model));self.d=mujoco.MjData(self.m)
        self.names=[self.m.joint(i).name for i in range(1,self.m.njnt)]
        self.js=np.array([i for i,n in enumerate(self.names) if any(k in n for k in ('hip','knee','ankle'))])
        self.qids=np.r_[np.arange(3),7+self.js];self.dofs=np.r_[np.arange(3),6+self.js]
        self.bodies=[self.m.body(s+'_ankle_pitch_link').id for s in ('left','right')]

    def solve(self,q,targets,previous=None):
        m,d=self.m,self.d;original=q[self.qids].copy();anchor=original if previous is None else previous
        target_r=[R.from_quat(t[3:]) for t in targets];jp=np.zeros((3,m.nv));jr=jp.copy()
        lo=np.r_[q[:3]-.5,m.jnt_range[1+self.js,0]+1e-5];hi=np.r_[q[:3]+.5,m.jnt_range[1+self.js,1]-1e-5]
        def evaluate(x,jac=False):
            d.qpos[:]=q;d.qpos[self.qids]=x;mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
            res=[];blocks=[]
            for i,b in enumerate(self.bodies):
                e=(target_r[i].inv()*R.from_matrix(d.xmat[b].reshape(3,3))).as_rotvec()
                res.extend([(d.xpos[b]-targets[i,:3])*100,e*100])
                if jac:
                    mujoco.mj_jacBody(m,d,jp,jr,b);ang=np.linalg.norm(e);k=skew(e);a=1/12 if ang<1e-4 else (1-.5*ang/np.tan(.5*ang))/ang**2
                    blocks.extend([jp[:,self.dofs]*100,(np.eye(3)-.5*k+a*k@k)@target_r[i].as_matrix().T@jr[:,self.dofs]*100])
            res.extend([(x[:3]-original[:3]),(x-anchor)*.01])
            if jac:
                root=np.zeros((3,len(x)));root[:,:3]=np.eye(3)
                return np.vstack([*blocks,root,np.eye(len(x))*.01])
            return np.concatenate(res)
        seeds=[anchor]
        if previous is None:
            for dz in (.1,.2,.35):
                seed=original.copy();seed[2]+=dz;seeds.append(seed)
        best=None
        for seed in seeds:
            opt=least_squares(evaluate,np.clip(seed,lo+1e-9,hi-1e-9),jac=lambda x:evaluate(x,True),bounds=(lo,hi),max_nfev=150,ftol=1e-9,xtol=1e-9,gtol=1e-9)
            cost=np.linalg.norm(opt.fun)
            if best is None or cost<best[0]:best=cost,opt.x
            if cost<.001:break
        result=q.copy();result[self.qids]=best[1];r=evaluate(best[1]);return result,best[1],dict(position_error_mm=[float(np.linalg.norm(r[i:i+3])*10) for i in (0,6)],rotation_error_deg=[float(np.rad2deg(np.linalg.norm(r[i:i+3])/100)) for i in (3,9)],pelvis_translation_m=(result[:3]-q[:3]).tolist())

    def solve_with_pelvis_rotation(self,q,targets,chest_rotation,previous=None):
        """Coupled lower-body fallback; keep the torso orientation target fixed."""
        m,d=self.m,self.d
        js=np.array([j for j,n in enumerate(self.names) if any(k in n for k in ('hip','knee','ankle','waist'))])
        r0=R.from_quat(q[[4,5,6,3]]);tb=m.body('torso_link').id;trs=[R.from_quat(t[3:]) for t in targets]
        x0=np.r_[q[:3],np.zeros(3),q[7+js]]
        lo=np.r_[q[:3]+[-.3,-.3,0],[-1.5]*3,m.jnt_range[1+js,0]+1e-5]
        hi=np.r_[q[:3]+[.3,.3,.6],[1.5]*3,m.jnt_range[1+js,1]-1e-5]
        if previous is not None:
            x0=np.r_[previous[:3],(r0.inv()*R.from_quat(previous[[4,5,6,3]])).as_rotvec(),previous[7+js]]
        def evaluate(x):
            d.qpos[:]=q;d.qpos[:3]=x[:3];d.qpos[3:7]=(r0*R.from_rotvec(x[3:6])).as_quat()[[3,0,1,2]];d.qpos[7+js]=x[6:]
            mujoco.mj_kinematics(m,d);res=[]
            for i,b in enumerate(self.bodies):
                res.extend([(d.xpos[b]-targets[i,:3])*700,(trs[i].inv()*R.from_matrix(d.xmat[b].reshape(3,3))).as_rotvec()*100])
            res.extend([(chest_rotation.inv()*R.from_matrix(d.xmat[tb].reshape(3,3))).as_rotvec()*100,x[:3]-q[:3],x[3:6],(x[6:]-x0[6:])*.01])
            return np.concatenate(res)
        opt=least_squares(evaluate,np.clip(x0,lo+1e-9,hi-1e-9),bounds=(lo,hi),max_nfev=120,ftol=1e-8,xtol=1e-8,gtol=1e-8)
        residual=evaluate(opt.x);result=d.qpos.copy()
        return result,dict(position_error_mm=[float(np.linalg.norm(residual[i:i+3])*1000/700) for i in (0,6)],rotation_error_deg=[float(np.rad2deg(np.linalg.norm(residual[i:i+3])/100)) for i in (3,9)],pelvis_translation_m=(result[:3]-q[:3]).tolist(),coupled_pelvis=True)
