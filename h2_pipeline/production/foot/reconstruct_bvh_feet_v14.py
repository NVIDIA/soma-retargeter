"""BVH-grounded, H2-limit-aware foot reference reconstruction.

Raw mapped BVH targets are immutable. Grounded goals are derived from the
frozen foot collision geometry, independently of candidate FK. Joint limits,
native URDF speeds and a dense ground-clearance check are hard constraints.
Source tracking error and old-reference admission remain separate reports.
"""
from pathlib import Path
import argparse,csv,json,time,xml.etree.ElementTree as ET
import numpy as np
from scipy import sparse
from scipy.spatial.transform import Rotation as R,Slerp
import osqp
from native_projection import BASE,KEY,FootModel,load,sample,log_jac,skew,digest,save,compare_geometry

class SourceFootProblem:
    def __init__(self, warm_path, floor=0., margin=.00005, *, key=KEY, source_path=None, trace_path=None):
        self.floor=float(floor);self.margin=float(margin)
        self.bundle=BASE/'artifacts/h2_bvh_gpu_staged_v5'
        self.fm=FootModel(self.bundle/'code/gear_sonic/data/assets/robot_description/mjcf/h2.xml');self.m=self.fm.m
        self.source=Path(source_path) if source_path else BASE/'artifacts/h2_bvh_gpu_refine131k_116014_v6/refined/csv'/f'{key}.csv'
        self.warm=Path(warm_path);self.header,self.raw,self.cols,self.original=load(self.source,self.fm)
        hh,self.wraw,_,self.q0=load(self.warm,self.fm);assert hh==self.header and self.q0.shape==self.original.shape
        self.n=len(self.q0);self.js=self.fm.legs;self.d=15;self.N=self.n*self.d
        self.old=np.arange(self.n)/120.;self.scales=np.r_[[.005]*3,[.03]*12]
        self.trace_path=Path(trace_path) if trace_path else Path(json.loads((BASE/'artifacts/h2_foot_trajectory_v10/source_foot_reference_diagnosis.json').read_text())['trace'])
        z=np.load(self.trace_path);assert len(z['targets'])==self.n and float(z['fps'])==120.
        ids=[z['names'].tolist().index(s+'Foot') for s in ('Left','Right')]
        self.bvh=z['targets'][:,ids].astype(float)
        self.fm.d.qpos[:]=self.q0[0];self.fm.mj.mj_kinematics(self.m,self.fm.d)
        local=[]
        for body,g in zip(self.fm.bodies,self.fm.geom):
            world=(self.fm.corner_signs*self.m.geom_size[g])@self.fm.d.geom_xmat[g].reshape(3,3).T+self.fm.d.geom_xpos[g]
            local.append((world-self.fm.d.xpos[body])@self.fm.d.xmat[body].reshape(3,3))
        self.local=np.array(local)
        tree=ET.parse(self.bundle/'code/gear_sonic/data/assets/robot_description/urdf/h2/h2.urdf')
        speed={j.get('name'):float(j.find('limit').get('velocity')) for j in tree.getroot().findall('joint') if j.find('limit') is not None}
        self.speeds=np.array([speed[name] for name in self.fm.names]);self.step=self.speeds[self.js]/120.
        self.qlo=self.m.jnt_range[1+self.js,0];self.qhi=self.m.jnt_range[1+self.js,1]
        self.low=np.c_[self.q0[:,:3]-[.1,.1,.05],np.tile(self.qlo,(self.n,1))]
        self.high=np.c_[self.q0[:,:3]+[.1,.1,.15],np.tile(self.qhi,(self.n,1))]
        self.x0=np.c_[self.q0[:,:3],self.q0[:,7+self.js]]
        dense=np.arange((self.n-1)*5+1)/600.
        _,_,height=self.fm.poses(sample(self.q0,dense))
        self.initial_lift=max(0.,self.floor+self.margin-float(height.min()))
        self.x0[:,2]+=self.initial_lift
        assert np.all(self.x0>=self.low-1e-8) and np.all(self.x0<=self.high+1e-8)
        self.obs=[]
        for rate in (120,50):
            times=np.arange(int(np.floor(self.old[-1]*rate+1e-8))+1)/rate
            pp,rr,lift=self.goals(times)
            for k,t in enumerate(times):
                l=min(int(np.floor(t*120+1e-9)),self.n-1);r=min(l+1,self.n-1);a=max(0.,t*120-l)
                deps=[(l,1-a)] if a<1e-8 else [(l,1-a),(r,a)]
                self.obs.append(dict(rate=rate,sample=k,time=float(t),l=l,r=r,a=a,deps=deps,p=pp[k],rot=rr[k],goal_lift=lift[k],weight=np.sqrt(120/rate)))
        self.cut_times=set()
        rr=[];cc=[];vv=[]
        for i in range(self.n-2):
            for axis in range(3):
                for f,c in [(i,1.),(i+1,-2.),(i+2,1.)]:rr.append(i*3+axis);cc.append(f*self.d+axis);vv.append(c)
        self.root_smooth=sparse.coo_matrix((vv,(rr,cc)),shape=((self.n-2)*3,self.N)).tocsc()

    def goals(self,times):
        p=np.stack([np.stack([np.interp(times,self.old,self.bvh[:,s,j]) for j in range(3)],1) for s in range(2)],1)
        r=np.stack([Slerp(self.old,R.from_quat(self.bvh[:,s,3:]))(times).as_matrix() for s in range(2)],1)
        corners=np.einsum('nsij,skj->nski',r,self.local)+p[:,:,None,:]
        lift=np.maximum(self.floor-corners[:,:,:,2].min(2),0.)
        p[:,:,2]+=lift
        return p,r,lift

    def assemble(self,x):
        q=self.q0.copy();q[:,:3]=x[:,:3];q[:,7+self.js]=x[:,3:];return q

    def at(self,q,o):
        l,r,a=o['l'],o['r'],o['a'];v=q[l]*(1-a)+q[r]*a
        ra=R.from_quat(q[l,[4,5,6,3]]);rb=R.from_quat(q[r,[4,5,6,3]])
        v[3:7]=(ra*R.from_rotvec((ra.inv()*rb).as_rotvec()*a)).as_quat()[[3,0,1,2]];return v

    def kinematics(self,q,o,jac=False):
        f=self.fm;f.d.qpos[:]=q;f.mj.mj_kinematics(self.m,f.d);f.mj.mj_comPos(self.m,f.d)
        residual=[];blocks=[];hs=[];hj=[]
        for side,b in enumerate(f.bodies):
            p=f.d.xpos[b];rot=f.d.xmat[b].reshape(3,3)
            e=(R.from_matrix(o['rot'][side]).inv()*R.from_matrix(rot)).as_rotvec()
            residual.extend([(p-o['p'][side])/.005,e/np.deg2rad(1.)])
            offsets=self.local[side]@rot.T;hs.append(p[2]+offsets[:,2])
            if jac:
                jp=np.zeros((3,self.m.nv));jr=jp.copy();f.mj.mj_jacBody(self.m,f.d,jp,jr,b)
                jp=jp[:,f.dofs];jr=jr[:,f.dofs]
                blocks.extend([jp/.005,log_jac(e)@o['rot'][side].T@jr/np.deg2rad(1.)])
                hj.append(np.array([jp[2]-(skew(offset)@jr)[2] for offset in offsets]))
        for side in range(2):
            if o['goal_lift'][side]>1e-8:
                corner=int(np.argmin(hs[side]))
                residual.append(np.array([(hs[side][corner]-(self.floor+.0001))/.0005]))
                if jac:blocks.append(hj[side][corner][None,:]/.0005)
        return np.concatenate(residual)*o['weight'],np.vstack(blocks)*o['weight'] if jac else None,np.array(hs),np.array(hj) if jac else None

    def linearize(self,x):
        q=self.assemble(x);return [(o,*self.kinematics(self.at(q,o),o,True)) for o in self.obs]

    def measure(self,x,dense=True):
        q=self.assemble(x);cost=0.
        for o in self.obs:
            e,_,_,_=self.kinematics(self.at(q,o),o);cost+=float(e@e)
        added=((x-self.x0)/self.scales).ravel()
        cost+=.1*float(np.sum((self.root_smooth@added)**2))+.00001*float(added@added)
        t=np.arange((self.n-1)*5+1)/600. if dense else self.old
        _,_,h=self.fm.poses(sample(q,t));minidx=np.unravel_index(h.argmin(),h.shape)
        _,_,lift=self.goals(t)
        contact_max=float(np.where(lift>1e-8,h,-np.inf).max()*1000)
        ratios=abs(np.diff(q[:,7:],axis=0))*120/self.speeds
        violation=max(float(np.max(self.low-x)),float(np.max(x-self.high)),0.)
        return dict(cost=cost,grounded_target_height_max_mm=contact_max,min_clearance_mm=float(h.min()*1000),speed_ratio=float(ratios.max()),box_violation=violation,feasible=bool(h.min()>=self.floor-1e-7 and ratios.max()<=1+1e-6 and violation<=1e-8),ground_peak=dict(sample=int(minidx[0]),rate=600 if dense else 120,side=int(minidx[1])),dense_violations=[float(t[k]) for k in np.flatnonzero(h.min(1)<self.floor+1e-5)[:64]])

    def add_ground_cuts(self, times):
        existing={round(o['time']*600) for o in self.obs}
        added=0
        for t in times:
            tick=round(t*600)
            if tick in existing:continue
            p,r,lift=self.goals(np.array([t]));l=min(int(np.floor(t*120+1e-9)),self.n-1);a=max(0.,t*120-l);rr=min(l+1,self.n-1)
            deps=[(l,1-a)] if a<1e-8 else [(l,1-a),(rr,a)]
            self.obs.append(dict(rate=600,sample=tick,time=t,l=l,r=rr,a=a,deps=deps,p=p[0],rot=r[0],goal_lift=lift[0],weight=.5))
            existing.add(tick);added+=1
        return added

    def qp(self,x,linear):
        jr=[];jc=[];jv=[];res=[];ar=[];ac=[];av=[];lo=[];hi=[]
        def constraint(entries,lower,upper):
            row=len(lo)
            for col,value in entries:
                if abs(value)>1e-14:ar.append(row);ac.append(int(col));av.append(float(value))
            lo.append(float(lower));hi.append(float(upper))
        for o,e,j,height,hjac in linear:
            offset=len(res);res.extend(e.tolist())
            for frame,weight in o['deps']:
                block=j*weight*self.scales
                rows,cols=np.nonzero(abs(block)>1e-14)
                jr.extend((rows+offset).tolist());jc.extend((cols+frame*self.d).tolist());jv.extend(block[rows,cols].tolist())
            for side in range(2):
                for corner in range(8):
                    if height[side,corner]>self.floor+.025:continue
                    entries=[]
                    for frame,weight in o['deps']:
                        entries.extend((frame*self.d+k,v) for k,v in enumerate(hjac[side,corner]*weight*self.scales/.005))
                    constraint(entries,(self.floor+self.margin-height[side,corner])/.005,np.inf)
            # At least one physical sole corner must reach each grounded BVH target.
            for side in range(2):
                if o['goal_lift'][side]<=1e-8:continue
                corner=int(np.argmin(height[side]));entries=[]
                for frame,weight in o['deps']:
                    entries.extend((frame*self.d+k,v) for k,v in enumerate(hjac[side,corner]*weight*self.scales/.005))
                constraint(entries,-np.inf,(self.floor+.00035-height[side,corner])/.005)
        for i in range(self.n):
            for j in range(self.d):constraint([(i*self.d+j,1.)],max((self.low[i,j]-x[i,j])/self.scales[j],-2.),min((self.high[i,j]-x[i,j])/self.scales[j],2.))
        for edge in range(self.n-1):
            for j in range(12):
                cap=self.step[j];diff=x[edge+1,3+j]-x[edge,3+j]
                entries=[(edge*self.d+3+j,-self.scales[3+j]/cap),((edge+1)*self.d+3+j,self.scales[3+j]/cap)]
                constraint(entries,(-cap-diff)/cap,(cap-diff)/cap)
        J=sparse.coo_matrix((jv,(jr,jc)),shape=(len(res),self.N)).tocsc();r=np.array(res)
        # Weak absolute anchor and root smoothing select among redundant leg/root fits.
        diagonal=np.full(self.N,.002);P=J.T@J+sparse.diags(diagonal,format='csc');g=np.asarray(J.T@r).ravel()+.00001*((x-self.x0)/self.scales).ravel()
        smoothing=.1*(self.root_smooth.T@self.root_smooth);P=P+smoothing;g+=np.asarray(smoothing@((x-self.x0)/self.scales).ravel()).ravel()
        A=sparse.coo_matrix((av,(ar,ac)),shape=(len(lo),self.N)).tocsc()
        solver=osqp.OSQP();solver.setup(P=sparse.triu(P).tocsc(),q=g,A=A,l=np.array(lo),u=np.array(hi),verbose=False,max_iter=15000,eps_abs=1e-6,eps_rel=1e-6,polishing=True)
        sol=solver.solve();status=str(sol.info.status)
        info=dict(status=status,iterations=int(sol.info.iter),primal_residual=float(sol.info.prim_res),dual_residual=float(sol.info.dual_res),variables=self.N,constraints=len(lo),nonzeros=A.nnz)
        usable=sol.x is not None and np.isfinite(sol.x).all() and status in ('solved','solved inaccurate','maximum iterations reached')
        return sol.x.reshape(self.n,self.d)*self.scales if usable else None,info

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--warm',type=Path,default=BASE/'artifacts/h2_foot_trajectory_v10/window_native_01'/f'{KEY}.csv');ap.add_argument('--iterations',type=int,default=24);ap.add_argument('--initial',type=Path);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    p=SourceFootProblem(a.warm);x=p.x0.copy()
    if a.initial:
        hh,_,_,initial=load(a.initial,p.fm);assert hh==p.header
        assert np.array_equal(initial[:,3:7],p.q0[:,3:7])
        x=np.c_[initial[:,:3],initial[:,7+p.js]]
    qq=p.assemble(x);tt=np.arange((p.n-1)*5+1)/600.;_,_,hh=p.fm.poses(sample(qq,tt));_,_,ll=p.goals(tt)
    active=(hh.min(1)<p.floor+p.margin)|np.any((ll>1e-8)&(hh>p.floor+.00035),axis=1)
    p.add_ground_cuts([float(t) for t in tt[active]])
    current=p.measure(x);assert current['feasible'],current
    pins={str(path):digest(path) for path in [p.source,p.warm,p.trace_path]};begin=time.time();history=[]
    save(a.out/'manifest.json',dict(reference_contract='h2_bvh_grounded_joint_feasible_v14',key=KEY,source_hashes=pins,source_fps=120,ground_plane_z=p.floor,ground_solve_margin_m=p.margin,initial_global_root_lift_m=p.initial_lift,code_sha256=digest(__file__),tracking_objective_scales=dict(position_m=.005,rotation_deg=1.),preserve_root_rotation=True,preserve_upper_joint_values=True,old_preservation_is_diagnostic_only=True,training_started=False,restoration_target_height_mm=.45))
    print('INITIAL',json.dumps(current),flush=True)
    for it in range(a.iterations):
        linear=p.linearize(x);delta,qp=p.qp(x,linear);winner=None;trials=[]
        if delta is not None:
            for alpha in (1.,.5,.25,.125,.0625,.03125,.015625):
                trial=np.clip(x+alpha*delta,p.low,p.high);score=p.measure(trial);trials.append(dict(alpha=alpha,**score))
                # Restore the exact dense contact constraint before minimizing
                # tracking cost. A tracking-only merit can reject every step
                # required to enter the hard contact-feasible set.
                restoring=current['grounded_target_height_max_mm']>.45
                contact_improves=score['grounded_target_height_max_mm']<current['grounded_target_height_max_mm']-1e-4
                tracking_improves=score['cost']<current['cost']-1e-5
                contact_preserved=score['grounded_target_height_max_mm']<=max(.45,current['grounded_target_height_max_mm'])+1e-6
                if score['feasible'] and ((restoring and contact_improves) or (not restoring and contact_preserved and tracking_improves)):
                    winner=(trial,score);break
        if winner:x,current=winner
        added=0
        # Add worst dense samples whose grounded BVH goal is not yet reached.
        qq=p.assemble(x);tt=np.arange((p.n-1)*5+1)/600.;_,_,hh=p.fm.poses(sample(qq,tt));_,_,ll=p.goals(tt)
        floating=np.where(ll>1e-8,hh-(p.floor+.0004),-np.inf).max(1)
        worst=np.argsort(floating)[::-1];contact_times=[float(tt[k]) for k in worst[:64] if floating[k]>0]
        added+=p.add_ground_cuts(contact_times)
        if winner is None and trials:
            added+=p.add_ground_cuts([t for row in trials for t in row['dense_violations']])
        row=dict(iteration=it+1,accepted=winner is not None,score=current,qp=qp,ground_cuts_added=added,seconds=time.time()-begin,trials=trials,ground_target_float_max_mm=float(np.where(ll>1e-8,hh,-np.inf).max()*1000));history.append(row);save(a.out/'history.json',history)
        q=p.assemble(x);data=p.wraw.copy();data[:,1:4]=q[:,:3];data[:,p.cols]=q[:,7:]
        path=a.out/f'{KEY}.csv';temp=path.with_suffix('.tmp')
        with temp.open('w',newline='') as f:w=csv.writer(f);w.writerow(p.header);w.writerows(data)
        temp.replace(path);print(json.dumps({k:v for k,v in row.items() if k!='trials'}),flush=True)
        if added:current=p.measure(x)
        if winner is None and not added:break
    assert all(digest(path)==v for path,v in pins.items())
    q=p.assemble(x);npz={};stats={}
    for rate in (120,50,600):
        ts=np.arange(int(np.floor(p.old[-1]*rate+1e-8))+1)/rate;pp,rr,lift=p.goals(ts);cp,cr,heights=p.fm.poses(sample(q,ts));ep=np.linalg.norm(cp-pp,axis=-1)*1000;er=(R.from_matrix(rr.reshape(-1,3,3)).inv()*R.from_matrix(cr.reshape(-1,3,3))).magnitude().reshape(-1,2)*180/np.pi
        npz.update({f'times_{rate}':ts,f'bvh_grounded_position_{rate}':pp,f'bvh_grounded_rotation_{rate}':rr,f'bvh_ground_lift_{rate}':lift,f'feasible_position_{rate}':cp,f'feasible_rotation_{rate}':cr,f'sole_clearance_{rate}':heights})
        stats[str(rate)]=dict(min_clearance_mm=float(heights.min()*1000),ground_lift_max_mm=float(lift.max()*1000),position_error_mm={s:dict(mean=float(ep[:,i].mean()),p95=float(np.percentile(ep[:,i],95)),max=float(ep[:,i].max())) for i,s in enumerate(('Left','Right'))},rotation_error_deg={s:dict(mean=float(er[:,i].mean()),p95=float(np.percentile(er[:,i],95)),max=float(er[:,i].max())) for i,s in enumerate(('Left','Right'))})
    np.savez_compressed(a.out/'reconstructed_reference.npz',**npz)
    report=dict(reference_contract='h2_bvh_grounded_joint_feasible_v14',key=KEY,candidate=str(path),candidate_sha256=digest(path),source_hashes=pins,score=current,source_goal_tracking=stats,legacy_comparison=compare_geometry(p.fm,p.original,q),old_admission_performed=False,training_started=False,seconds=time.time()-begin)
    save(a.out/'complete.json',report);print('FINAL',json.dumps(report),flush=True)

if __name__=='__main__':main()
