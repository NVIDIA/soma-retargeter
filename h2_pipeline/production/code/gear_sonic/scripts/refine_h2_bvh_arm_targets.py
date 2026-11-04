"""Kinematic arm stage for fresh BVH retargets; no training-PKL modification.

Uses original BVH bone directions, H2 joint limits and bounded angular speed.
Free base, torso, legs and source frame IDs are bit-preserved in the CSV.
"""
import argparse,csv,json,hashlib,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np,mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R


def skew(x):
    a,b,c=x;return np.array([[0,-c,b],[c,0,-a],[-b,a,0]])


def arm_joint_bounds(model, joint_ids, margin_deg=0.):
    """Keep shoulder/wrist coordinates inside physical limits by a hard margin.

    Elbow bounds stay unchanged so a straight elbow remains representable.
    This is a reference clearance, not a dynamic safety guarantee.
    """
    if not np.isfinite(margin_deg) or margin_deg < 0:
        raise ValueError('Joint margin must be finite and nonnegative')
    limits = model.jnt_range[joint_ids].copy()
    margins = np.array([np.deg2rad(margin_deg) if any(k in model.joint(int(j)).name for k in ('shoulder', 'wrist')) else 0. for j in joint_ids])
    lo = limits[:, 0] + margins + 1e-5
    hi = limits[:, 1] - margins - 1e-5
    if np.any(lo >= hi):
        raise ValueError('Joint margin leaves no feasible interval')
    return lo, hi


class ArmSolver:
    def __init__(self,model,velocity_cap=8.,geometry_weight=10.,wrist_weight=.03,elbow_weight=.1,continuity_weight=.02,canonical_weight=.01,multistart=0,bound_speed=True,forearm_weight=None,elbow_flex_weight=0.,joint_limit_margin_deg=2.):
        self.geometry_weight=geometry_weight;self.direction_weights=np.array([geometry_weight,geometry_weight if forearm_weight is None else forearm_weight]);self.wrist_weight=wrist_weight;self.elbow_weight=elbow_weight
        self.joint_limit_margin_deg=joint_limit_margin_deg;self.elbow_flex_weight=elbow_flex_weight;self.bound_speed=bound_speed;self.continuity_weight=continuity_weight;self.canonical_weight=canonical_weight;self.multistart=multistart
        self.m=mujoco.MjModel.from_xml_path(str(model));self.d=mujoco.MjData(self.m)
        self.names=[self.m.joint(i).name for i in range(1,self.m.njnt)]
        if not np.isfinite(velocity_cap) or velocity_cap<=0:raise ValueError('Positive finite angular speed cap required')
        urdf=Path(model).resolve().parent.parent/'urdf/h2/h2.urdf'
        limits={j.get('name'):float(j.find('limit').get('velocity')) for j in ET.parse(urdf).getroot().findall('joint') if j.find('limit') is not None}
        self.velocity_limits=np.array([min(velocity_cap,limits[n]) for n in self.names])
    def solve(self,q,side,human_directions,wrist_rotation,previous,canonical,dt,elbow_rotation=None):
        m,d=self.m,self.d;js=np.array([i for i,n in enumerate(self.names) if n.startswith(side.lower()+'_') and any(s in n for s in ('shoulder','elbow','wrist'))]);dofs=6+js
        bodies=[m.body(side.lower()+'_'+s+'_link').id for s in ('shoulder_roll','elbow','wrist_yaw')]
        current=q[7+js].copy();anchor=canonical if previous is None else previous
        lo,hi=arm_joint_bounds(m,1+js,self.joint_limit_margin_deg)
        if previous is not None and self.bound_speed:
            lo=np.maximum(lo,previous-self.velocity_limits[js]*dt);hi=np.minimum(hi,previous+self.velocity_limits[js]*dt)
        jacp=np.zeros((3,m.nv));jacr=jacp.copy()
        def evaluate(x,jac=False):
            d.qpos[:]=q;d.qpos[7+js]=x;mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
            vectors=np.diff(d.xpos[bodies],axis=0);length=np.linalg.norm(vectors,axis=1);u=vectors/length[:,None]
            wr=R.from_matrix(d.xmat[bodies[-1]].reshape(3,3));e=(wrist_rotation.inv()*wr).as_rotvec()
            ee=np.zeros(3) if elbow_rotation is None else (elbow_rotation.inv()*R.from_matrix(d.xmat[bodies[1]].reshape(3,3))).as_rotvec()
            flex=np.arccos(np.clip(u[0]@u[1],-1,1));human_flex=np.arccos(np.clip(human_directions[0]@human_directions[1],-1,1))
            residual=np.r_[((u-human_directions)*self.direction_weights[:,None]).ravel(),e*self.wrist_weight,ee*self.elbow_weight,(flex-human_flex)*self.elbow_flex_weight,(x-anchor)*self.continuity_weight,(x-canonical)*self.canonical_weight]
            if not jac:return residual
            jp=[]
            for body in bodies:
                mujoco.mj_jacBody(m,d,jacp,jacr,body);jp.append(jacp[:,dofs].copy())
            out=[]
            for i in range(2):out.append((np.eye(3)-np.outer(u[i],u[i]))/length[i]@(jp[i+1]-jp[i])*self.direction_weights[i])
            angle=np.linalg.norm(e);k=skew(e);a=1/12 if angle<1e-4 else (1-.5*angle/np.tan(.5*angle))/angle**2
            out.append((np.eye(3)-.5*k+a*k@k)@wrist_rotation.as_matrix().T@jacr[:,dofs]*self.wrist_weight)
            if elbow_rotation is None:out.append(np.zeros((3,len(js))))
            else:
                mujoco.mj_jacBody(m,d,jacp,jacr,bodies[1]);ang=np.linalg.norm(ee);kk=skew(ee);aa=1/12 if ang<1e-4 else (1-.5*ang/np.tan(.5*ang))/ang**2
                out.append((np.eye(3)-.5*kk+aa*kk@kk)@elbow_rotation.as_matrix().T@jacr[:,dofs]*self.elbow_weight)
            flex_jac=-(u[1]@(out[0]/self.direction_weights[0])+u[0]@(out[1]/self.direction_weights[1]))/max(np.sin(flex),1e-7)*self.elbow_flex_weight
            out.append(flex_jac[None,:])
            out.extend([np.eye(len(js))*self.continuity_weight,np.eye(len(js))*self.canonical_weight]);return np.vstack(out)
        seeds=[anchor]
        if previous is None or not self.bound_speed:
            seeds += [canonical,current,np.array([-1.5,0,0,1.5,0,0,0]),np.array([0,1.2 if side=='Left' else -1.2,0,1.5,0,0,0])]
        if (previous is None or not self.bound_speed) and self.multistart:
            seeds.extend(np.random.default_rng(17).uniform(lo,hi,(self.multistart,len(js))))
        best=None
        for seed in seeds:
            sol=least_squares(evaluate,np.clip(seed,lo+1e-10,hi-1e-10),jac=lambda x:evaluate(x,True),bounds=(lo,hi),max_nfev=140,ftol=1e-8,xtol=1e-8,gtol=1e-8)
            cost=np.linalg.norm(evaluate(sol.x))
            if best is None or cost<best[0]:best=(cost,sol.x)
            if cost<.002:break
        result=q.copy();result[7+js]=best[1]
        return result,best[1],float(np.linalg.norm(evaluate(best[1])[:6].reshape(2,3)/self.direction_weights[:,None]))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--input',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--model',type=Path,required=True);ap.add_argument('--reference',type=Path,required=True);ap.add_argument('--reverse',action='store_true');ap.add_argument('--max-joint-speed-rad-s',type=float,default=8.);ap.add_argument('--geometry-weight',type=float,default=10.);ap.add_argument('--wrist-weight',type=float,default=.03);ap.add_argument('--elbow-weight',type=float,default=.1);ap.add_argument('--continuity-weight',type=float,default=.02);ap.add_argument('--canonical-weight',type=float,default=.01);ap.add_argument('--multistart',type=int,default=0);ap.add_argument('--keys',nargs='+');ap.add_argument('--unbounded-speed',action='store_true');ap.add_argument('--joint-limit-margin-deg',type=float,default=2.);a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);(a.out/'csv').mkdir();solver=ArmSolver(a.model,a.max_joint_speed_rad_s,a.geometry_weight,a.wrist_weight,a.elbow_weight,a.continuity_weight,a.canonical_weight,a.multistart,not a.unbounded_speed,joint_limit_margin_deg=a.joint_limit_margin_deg);canonical=np.load(a.reference)['qpos'];reports=[]
    (a.out/'runner.py').write_bytes(Path(__file__).read_bytes())
    for path in sorted((a.input/'csv').glob('*.csv')):
        if a.keys and path.stem not in a.keys:continue
        with path.open() as f:rd=csv.reader(f);header=next(rd);raw=np.array(list(rd),float)
        clean=[s.removesuffix('_dof') for s in header];cols=[clean.index(n) for n in solver.names];data=raw.copy()
        trace=np.load(a.input/(path.stem+'_trace.npz'));source=trace['source_world'];sn=trace['source_names'].tolist();tn=trace['names'].tolist();fps=float(trace['fps'])
        if len(raw)!=len(source):raise ValueError('Source timebase mismatch')
        prev={'Left':None,'Right':None};errors=[]
        for i in (range(len(raw)-1,-1,-1) if a.reverse else range(len(raw))):
            row=raw[i]
            q=np.r_[row[1:4],R.from_euler('xyz',row[4:7]).as_quat()[[3,0,1,2]],row[cols]]
            for side in ('Left','Right'):
                hp=source[i,[sn.index(side+s) for s in ('Arm','ForeArm','Hand')],:3];hv=np.diff(hp,axis=0);hv/=np.linalg.norm(hv,axis=1)[:,None]
                js=np.array([j for j,n in enumerate(solver.names) if n.startswith(side.lower()+'_') and any(x in n for x in ('shoulder','elbow','wrist'))])
                wrist=R.from_quat(trace['targets'][i,tn.index(side+'Hand'),3:])
                q,prev[side],err=solver.solve(q,side,hv,wrist,prev[side],canonical[7+js],1/fps,R.from_quat(trace['targets'][i,tn.index(side+'ForeArm'),3:]));errors.append(err)
            data[i,cols]=q[7:]
        keep=[i for i,h in enumerate(clean) if not any(s in h for s in ('shoulder','elbow','wrist'))]
        if not np.array_equal(data[:,keep],raw[:,keep]):raise AssertionError('Lower body or timebase changed')
        with (a.out/'csv'/path.name).open('w') as f:
            wr=csv.writer(f);wr.writerow(header);wr.writerows(data)
        row=dict(key=path.stem,frames=len(raw),source_csv_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),max_direction_residual=max(errors),non_arm_and_frame_columns_bit_identical=True)
        reports.append(row);print(json.dumps(row),flush=True)
    (a.out/'report.json').write_text(json.dumps(dict(scope=__doc__,joint_limit_margin_deg=a.joint_limit_margin_deg,solver_weights=dict(geometry=a.geometry_weight,wrist=a.wrist_weight,elbow=a.elbow_weight,continuity=a.continuity_weight,canonical=a.canonical_weight),multistart=a.multistart,speed_bound_enabled=not a.unbounded_speed,reverse_optimization=a.reverse,max_joint_speed_rad_s=a.max_joint_speed_rad_s,samples=reports,dynamic_retarget=False,published=False,training_started=False),indent=2)+'\n')
if __name__=='__main__':main()
