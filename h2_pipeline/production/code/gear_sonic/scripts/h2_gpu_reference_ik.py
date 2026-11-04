"""CUDA reference IK candidate solver; independent admission is mandatory.

MJCF supplies constants only. FK, analytic Jacobians and LM solves run on CUDA.
This is kinematic optimization, not inverse dynamics or policy validation.
"""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
import torch
from scipy.spatial.transform import Rotation as R
from gear_sonic.scripts.h2_waist_gpu_ik import skew, log_rotation


class GPUReferenceIK:
    def __init__(self, model, device='cuda:0'):
        if not str(device).startswith('cuda') or not torch.cuda.is_available():
            raise RuntimeError('CUDA required; CPU optimization fallback is disabled')
        self.device = device
        self.m = m = mujoco.MjModel.from_xml_path(str(model))
        if m.nq != 38 or m.nv != 37 or m.body_jntnum[1] != 1:
            raise ValueError('Expected H2 free base and 31 hinge joints')
        self.t = lambda x: torch.as_tensor(x, device=device, dtype=torch.float64)
        self.eye = self.t(np.eye(3))
        self.names = [m.joint(i).name for i in range(1, m.njnt)]
        self.body_col = {int(m.jnt_bodyid[j]): j-1 for j in range(1,m.njnt)}
        self.nodes = []
        for b in range(2,m.nbody):
            if m.body_jntnum[b] > 1: raise ValueError('Multiple hinges on a body unsupported')
            col = self.body_col.get(b)
            j = col+1 if col is not None else None
            self.nodes.append((b,int(m.body_parentid[b]),self.t(m.body_pos[b]),
                self.t(R.from_quat(m.body_quat[b][[1,2,3,0]]).as_matrix()),col,
                self.t(m.jnt_pos[j]) if j else None,self.t(m.jnt_axis[j]) if j else None,
                float(m.qpos0[m.jnt_qposadr[j]]) if j else 0.))
        masks=[]
        for b in range(m.nbody):
            ancestors=set();node=b
            while node>1: ancestors.add(node);node=int(m.body_parentid[node])
            masks.append([int(m.jnt_bodyid[j]) in ancestors for j in range(1,m.njnt)])
        self.masks=self.t(masks)
        margin=np.array([np.deg2rad(2.) if any(s in n for s in ('shoulder','wrist')) else 0. for n in self.names])
        self.lo=self.t(m.jnt_range[1:,0]+margin+1e-5)
        self.hi=self.t(m.jnt_range[1:,1]-margin-1e-5)
        urdf=Path(model).resolve().parent.parent/'urdf/h2/h2.urdf'
        speeds={j.get('name'):float(j.find('limit').get('velocity')) for j in ET.parse(urdf).getroot().findall('joint') if j.find('limit') is not None}
        self.speeds=self.t([speeds[n] for n in self.names])
        self.arm_weights=self.t([2.,2.]);self.arm_indices={};self.arm_nodes={}
        for side in ('Left','Right'):
            js=[i for i,n in enumerate(self.names) if n.startswith(side.lower()+'_') and any(k in n for k in ('shoulder','elbow','wrist'))]
            bodies=[m.body(side.lower()+'_'+n+'_link').id for n in ('shoulder_roll','elbow','wrist_yaw')]
            self.arm_indices[side]=(torch.tensor(js,device=device,dtype=torch.long),torch.tensor(bodies,device=device,dtype=torch.long),bodies[-1])
            needed=set();node=bodies[-1]
            while node>1:needed.add(node);node=int(m.body_parentid[node])
            self.arm_nodes[side]=[row for row in self.nodes if row[0] in needed]

    def fk(self, root, root_r, angles, nodes=None):
        p={0:torch.zeros_like(root),1:root};r={0:self.eye.expand(len(root),3,3),1:root_r}
        zero=torch.zeros_like(root);axes=[zero]*31;pivots=[zero]*31
        for b,parent,bpos,base,col,jpos,axis,ref in (self.nodes if nodes is None else nodes):
            rb=r[parent]@base;origin=p[parent]+(r[parent]@bpos[:,None]).squeeze(-1)
            if col is None:p[b]=origin;r[b]=rb;continue
            a=angles[:,col]-ref;k=skew(axis)
            hinge=self.eye+torch.sin(a)[:,None,None]*k+(1-torch.cos(a))[:,None,None]*(k@k)
            pivot=origin+(rb@jpos[:,None]).squeeze(-1)
            axes[col]=(rb@axis[:,None]).squeeze(-1);pivots[col]=pivot
            r[b]=rb@hinge;p[b]=pivot-(r[b]@jpos[:,None]).squeeze(-1)
        return torch.stack([p.get(b,zero) for b in range(self.m.nbody)],1),torch.stack([r.get(b,root_r) for b in range(self.m.nbody)],1),torch.stack(axes,1),torch.stack(pivots,1)

    def body_jac(self, p, axes, pivots, bodies, js):
        axis=axes[:,None,js]*self.masks[bodies][:,js][None,:,:,None]
        jp=torch.cross(axis,p[:,bodies,None]-pivots[:,None,js],dim=-1).transpose(-1,-2)
        return jp,axis.transpose(-1,-2)

    def orientation(self, actual, target, jr):
        e=log_rotation(target.transpose(-1,-2)@actual);k=skew(e)
        theta=torch.linalg.vector_norm(e,dim=-1)
        c=(1-.5*theta/torch.tan(.5*theta).clamp(min=1e-15))/theta.square().clamp(min=1e-15)
        c=torch.where(theta<1e-4,torch.full_like(c,1/12),c)
        jac=(self.eye-.5*k+c[...,None,None]*(k@k))@target.transpose(-1,-2)@jr
        return e,jac

    def arm_geometry(self, q, root_r, side, x, directions, wrist, sequence=False):
        js,bodies,last_body=self.arm_indices[side]
        angles=q[:,7:].clone();angles[:,js]=x
        p,r,axes,pivots=self.fk(q[:,:3],root_r,angles,self.arm_nodes[side])
        jp,jr=self.body_jac(p,axes,pivots,bodies,js)
        v=p[:,bodies[1:]]-p[:,bodies[:-1]];length=torch.linalg.vector_norm(v,dim=-1)
        u=v/length[:,:,None];dv=jp[:,1:]-jp[:,:-1]
        du=((self.eye-u[:,:,:,None]*u[:,:,None,:])/length[:,:,None,None])@dv
        e,je=self.orientation(r[:,last_body],wrist,jr[:,-1])
        priority=getattr(self,'sequence_priority',(2.,2.,50.)) if sequence else (2.,2.,50.)
        weights=self.t(priority[:2]) if sequence else self.arm_weights;ww=priority[2]
        res=[((u-directions)*weights[None,:,None]).flatten(1),e*ww]
        jac=[(du*weights[None,:,None,None]).flatten(1,2),je*ww]
        if sequence:
            dot=(u[:,0]*u[:,1]).sum(-1).clamp(-1+1e-14,1-1e-14)
            human=torch.acos((directions[:,0]*directions[:,1]).sum(-1).clamp(-1,1))
            flex=torch.acos(dot);weight=(human<=np.deg2rad(15)).to(x.dtype)*getattr(self,'flex_weight',6.)
            jf=-(u[:,1,None,:]@du[:,0]+u[:,0,None,:]@du[:,1]).squeeze(1)/torch.sin(flex).clamp(min=1e-7)[:,None]
            res.append(((flex-human)*weight)[:,None]);jac.append((jf*weight[:,None])[:,None,:])
            peak=getattr(self,'sequence_peak',None)
            if peak is not None:
                upper,forearm,wrist_limit,direction_weight,wrist_weight=peak
                diff=u-directions;norm=diff.norm(dim=-1);threshold=self.t([2*np.sin(np.deg2rad(upper)/2),2*np.sin(np.deg2rad(forearm)/2)])
                factor=(1-threshold/norm.clamp(min=1e-12)).clamp(min=0)
                scale=self.eye*factor[:,:,None,None]+threshold[None,:,None,None]*(diff[:,:,:,None]*diff[:,:,None,:])/norm.clamp(min=1e-12)[:,:,None,None]**3
                scale=torch.where((norm>threshold)[:,:,None,None],scale,torch.zeros_like(scale))
                res.append((diff*factor[:,:,None]).flatten(1)*direction_weight);jac.append((scale@du).flatten(1,2)*direction_weight)
                n=e.norm(dim=-1);threshold=np.deg2rad(wrist_limit);factor=(1-threshold/n.clamp(min=1e-12)).clamp(min=0)
                scale=self.eye*factor[:,None,None]+threshold*(e[:,:,None]*e[:,None,:])/n.clamp(min=1e-12)[:,None,None]**3
                scale=torch.where((n>threshold)[:,None,None],scale,torch.zeros_like(scale))
                res.append(e*factor[:,None]*wrist_weight);jac.append((scale@je)*wrist_weight)
        return torch.cat(res,1),torch.cat(jac,1)

    @torch.inference_mode()
    def lm(self, x, lo, hi, evaluate, iterations=80, step_cap=.3):
        x=x.clamp(lo,hi);eye=torch.eye(x.shape[1],dtype=x.dtype,device=x.device)
        damping=torch.full((len(x),),.01,device=x.device,dtype=x.dtype)
        for _ in range(iterations):
            residual,jac=evaluate(x);jt=jac.transpose(-1,-2)
            gradient=(jt@residual[:,:,None]).squeeze(-1)
            free=~(((x<=lo+1e-8)&(gradient>0))|((x>=hi-1e-8)&(gradient<0)))
            normal=(jt@jac)*free[:,:,None]*free[:,None,:];diagonal=normal.diagonal(dim1=-2,dim2=-1).clamp(min=1.)
            step=torch.linalg.solve_ex(normal+torch.diag_embed(damping[:,None]*diagonal)+eye*1e-10,-(gradient*free)[:,:,None],check_errors=False).result.squeeze(-1).clamp(-step_cap,step_cap)
            trial=(x+step).clamp(lo,hi);rr,_=evaluate(trial)
            take=rr.square().sum(1)<=residual.square().sum(1)+1e-15
            x=torch.where(take[:,None],trial,x)
            damping=torch.where(take,damping*.5,damping*4).clamp(1e-9,1e9)
        return x

    @torch.inference_mode()
    def solve_chunk(self, q_np, targets_np, target_names, directions_np, iterations=100, canonical=None, seed_mode="canonical", skip_lower=False, lower_only=False):
        q=self.t(q_np);original_q=q.clone();root_r=self.t(R.from_quat(q_np[:,[4,5,6,3]]).as_matrix())
        targets=self.t(targets_np);tr=self.t(R.from_quat(targets_np[:,:,3:].reshape(-1,4)).as_matrix().reshape(len(q),-1,3,3))
        if not skip_lower:
            js=[i for i,n in enumerate(self.names) if any(k in n for k in ('hip','knee','ankle'))]
            bodies=[self.m.body(n).id for n in ('left_ankle_pitch_link','right_ankle_pitch_link')]
            ti=[target_names.index(n) for n in ('LeftFoot','RightFoot')]
            tp=targets[:,ti[:2],:3];rr=tr[:,ti];original=torch.cat([q[:,:3],q[:,7+np.array(js)]],1)
            def lower(x):
                angles=q[:,7:].clone();angles[:,js]=x[:,3:]
                p,r,axes,pivots=self.fk(x[:,:3],root_r,angles)
                jp,jr=self.body_jac(p,axes,pivots,bodies,js)
                jp=torch.cat([self.eye.expand(len(x),2,3,3),jp],-1)
                jr=torch.cat([torch.zeros((len(x),2,3,3),device=x.device,dtype=x.dtype),jr],-1)
                e,je=self.orientation(r[:,bodies],rr,jr)
                res=torch.cat([(p[:,bodies[:2]]-tp).flatten(1)*100,e.flatten(1)*100,(x[:,:3]-original[:,:3]),(x-original)*.01],1)
                root=torch.zeros((len(x),3,len(js)+3),device=x.device,dtype=x.dtype);root[:,:,:3]=self.eye
                eye=torch.eye(len(js)+3,device=x.device,dtype=x.dtype).expand(len(x),-1,-1)
                jac=torch.cat([jp[:,:2].flatten(1,2)*100,je.flatten(1,2)*100,root,eye*.01],1)
                return res,jac
            lo=torch.cat([q[:,:3]-.5,self.lo[js].expand(len(q),-1)],1)
            hi=torch.cat([q[:,:3]+.5,self.hi[js].expand(len(q),-1)],1)
            x=self.lm(original,lo,hi,lower,iterations);best=lower(x)[0].square().sum(1)
            for dz in (.1,.2,.35):
                seed=original.clone();seed[:,2]+=dz;trial=self.lm(seed,lo,hi,lower,iterations);cost=lower(trial)[0].square().sum(1)
                x=torch.where((cost<best)[:,None],trial,x);best=torch.minimum(best,cost)
            q[:,:3]=x[:,:3];q[:,7+np.array(js)]=x[:,3:]
            residual,_=lower(x);bad=(residual[:,:6].reshape(-1,2,3).norm(dim=-1).max(1).values>.1)|(residual[:,6:12].reshape(-1,2,3).norm(dim=-1).max(1).values>np.deg2rad(.1)*100)
            if bool(bad.any()):
                corrected,corrected_r=self.lower_coupled(original_q[bad],q[bad],root_r[bad],tp[bad],tr[bad][:,[target_names.index(n) for n in ('LeftFoot','RightFoot','Chest')]],iterations)
                q[bad]=corrected;root_r[bad]=corrected_r
        if lower_only:return q.cpu().numpy()
        for side in ('Left','Right'):
            js=[i for i,n in enumerate(self.names) if n.startswith(side.lower()+'_') and any(k in n for k in ('shoulder','elbow','wrist'))]
            dirs=self.t(directions_np[side]);wrist=tr[:,target_names.index(side+'Hand')]
            def arm(x): return self.arm_geometry(q,root_r,side,x,dirs,wrist)
            initial=q[:,7+np.array(js)].clone() if seed_mode=='raw' else self.t(self.m.qpos0[7+np.array(js)] if canonical is None else canonical[7+np.array(js)]).expand(len(q),-1)
            x=self.lm(initial,self.lo[js],self.hi[js],arm,iterations)
            q[:,7+np.array(js)]=x
        if not torch.isfinite(q).all(): raise FloatingPointError('Nonfinite GPU IK candidate')
        return q.cpu().numpy()

    @torch.inference_mode()
    def lower_coupled(self,original,initial,base_r,target_p,target_r,iterations):
        js=[i for i,n in enumerate(self.names) if any(k in n for k in ('hip','knee','ankle','waist'))]
        bodies=[self.m.body(n).id for n in ('left_ankle_pitch_link','right_ankle_pitch_link','torso_link')]
        def rotation(w):
            theta=torch.linalg.vector_norm(w,dim=-1);k=skew(w);t2=theta.square().clamp(min=1e-20)
            aa=torch.where(theta<1e-5,1-theta.square()/6,torch.sin(theta)/theta.clamp(min=1e-20))
            bb=torch.where(theta<1e-5,.5-theta.square()/24,(1-torch.cos(theta))/t2)
            cc=torch.where(theta<1e-5,1/6-theta.square()/120,(theta-torch.sin(theta))/(t2*theta).clamp(min=1e-20))
            return base_r@(self.eye+aa[:,None,None]*k+bb[:,None,None]*(k@k)),base_r@(self.eye+bb[:,None,None]*k+cc[:,None,None]*(k@k))
        x0=torch.cat([initial[:,:3],torch.zeros_like(initial[:,:3]),initial[:,7+np.array(js)]],1)
        def evaluate(x):
            rr,world_axes=rotation(x[:,3:6]);angles=original[:,7:].clone();angles[:,js]=x[:,6:]
            p,r,axes,pivots=self.fk(x[:,:3],rr,angles);jp,jr=self.body_jac(p,axes,pivots,bodies,js)
            root_jp=-skew(p[:,bodies]-x[:,None,:3])@world_axes[:,None]
            jp=torch.cat([self.eye.expand(len(x),3,3,3),root_jp,jp],-1)
            jr=torch.cat([torch.zeros((len(x),3,3,3),device=self.device,dtype=x.dtype),world_axes[:,None].expand(-1,3,-1,-1),jr],-1)
            e,je=self.orientation(r[:,bodies],target_r,jr)
            eye=torch.eye(x.shape[1],device=self.device,dtype=x.dtype).expand(len(x),-1,-1)
            residual=torch.cat([(p[:,bodies[:2]]-target_p).flatten(1)*700,e.flatten(1)*100,x[:,:3]-original[:,:3],x[:,3:6],(x[:,6:]-x0[:,6:])*.01],1)
            jac=torch.cat([jp[:,:2].flatten(1,2)*700,je.flatten(1,2)*100,eye[:,:3],eye[:,3:6],eye[:,6:]*.01],1)
            return residual,jac
        lo=torch.cat([original[:,:3]+self.t([-.3,-.3,0]),torch.full_like(original[:,:3],-1.5),self.lo[js].expand(len(original),-1)],1)
        hi=torch.cat([original[:,:3]+self.t([.3,.3,.6]),torch.full_like(original[:,:3],1.5),self.hi[js].expand(len(original),-1)],1)
        x=self.lm(x0,lo,hi,evaluate,iterations);rr,_=rotation(x[:,3:6]);answer=original.clone();answer[:,:3]=x[:,:3];answer[:,7+np.array(js)]=x[:,6:]
        answer[:,3:7]=self.t(R.from_matrix(rr.cpu().numpy()).as_quat()[:,[3,0,1,2]])
        return answer,rr

    @torch.inference_mode()
    def sequence_arm(self, q_np, side, directions, wrist_rotation, fps, iterations=100, starts=(0,)):
        """Whole-clip GPU Gauss-Newton with matrix-free temporal normal equations.

        Uses the CPU sequence objective's direction/wrist/flex/velocity weights.
        PCG and line search stay on CUDA; no SciPy optimizer is called.
        """
        q=self.t(q_np);root_r=self.t(R.from_quat(q_np[:,[4,5,6,3]]).as_matrix())
        js=[i for i,n in enumerate(self.names) if n.startswith(side.lower()+'_') and any(k in n for k in ('shoulder','elbow','wrist'))]
        anchor=q[:,7+np.array(js)].clone();x=anchor.clamp(self.lo[js],self.hi[js]);dirs=self.t(directions);wrist=self.t(wrist_rotation)
        caps=self.speeds[js]*.98/fps;vw=1500.;sw=.2;aw=.003;eye=torch.eye(7,device=self.device,dtype=x.dtype)
        edge=torch.ones((len(q)-1,1),device=self.device,dtype=x.dtype)
        for start in starts:
            if start>0:edge[start-1]=0
        triple=edge[:-1]*edge[1:]
        def temporal(v):
            diff=(v[1:]-v[:-1])*edge;over=(abs(diff)-caps).clamp(min=0)*torch.sign(diff)
            second=(v[2:]-2*v[1:-1]+v[:-2])*triple
            return diff,over,second
        def transpose(d1,d2):
            d1=d1*edge;d2=d2*triple
            out=torch.zeros_like(x);out[:-1]-=d1;out[1:]+=d1
            out[:-2]+=d2;out[1:-1]-=2*d2;out[2:]+=d2
            return out
        def objective(v):
            res,jac=self.arm_geometry(q,root_r,side,v,dirs,wrist,True)
            diff,over,second=temporal(v)
            cost=sums(res.square().sum(1)+aw**2*(v-anchor).square().sum(1))+sums(sw**2*second.square().sum(1),labels[:-2])+sums(vw**2*over.square().sum(1),labels[:-1])
            return cost,res,jac,diff,over,second
        labels=torch.empty(len(q),device=self.device,dtype=torch.long)
        for label,(begin,end) in enumerate(zip(starts,list(starts[1:])+[len(q)])):labels[begin:end]=label
        def sums(values,idx=labels):
            out=torch.zeros(len(starts),device=self.device,dtype=x.dtype)
            return out.index_add_(0,idx,values)
        damping=torch.full((len(starts),),.01,device=self.device,dtype=x.dtype);history=[]
        for iteration in range(iterations):
            cost,res,jac,diff,over,second=objective(x);jt=jac.transpose(-1,-2);normal=jt@jac
            active=(abs(diff)>caps).to(x.dtype)*vw**2
            gradient=(jt@res[:,:,None]).squeeze(-1)+aw**2*(x-anchor)+transpose(over*vw**2,second*sw**2)
            free=~(((x<=self.lo[js]+1e-8)&(gradient>0))|((x>=self.hi[js]-1e-8)&(gradient<0)))
            gradient=gradient*free
            diag=normal.diagonal(dim1=-2,dim2=-1).clamp(min=1.)*damping[labels,None]+aw**2
            prediag=diag.clone();prediag[:-1]+=active;prediag[1:]+=active
            prediag[:-2]+=sw**2*triple;prediag[1:-1]+=4*sw**2*triple;prediag[2:]+=sw**2*triple
            chol=torch.linalg.cholesky(normal+torch.diag_embed(prediag)+eye*1e-9)
            def matvec(v):
                vv=v*free
                return ((normal@vv[:,:,None]).squeeze(-1)+diag*vv+transpose((vv[1:]-vv[:-1])*active,(vv[2:]-2*vv[1:-1]+vv[:-2])*sw**2))*free+v*(~free)
            def precondition(v):return torch.cholesky_solve((v*free)[:,:,None],chol).squeeze(-1)*free
            step=torch.zeros_like(x);r=-gradient;z=precondition(r);direction=z.clone();rz=sums((r*z).sum(1))
            for _ in range(80):
                hd=matvec(direction);alpha=rz/sums((direction*hd).sum(1)).clamp(min=1e-30)
                step+=alpha[labels,None]*direction;r-=alpha[labels,None]*hd;z=precondition(r);newrz=sums((r*z).sum(1))
                beta=newrz/rz.clamp(min=1e-30);direction=z+beta[labels,None]*direction;rz=newrz
            step=step.clamp(-.3,.3);accepted=torch.zeros(len(starts),device=self.device,dtype=torch.bool);base=x.clone()
            for scale in (1.,.5,.25,.125,.0625):
                trial=(base+scale*step).clamp(self.lo[js],self.hi[js]);newcost,*_=objective(trial)
                take=(newcost<=cost)&~accepted;x=torch.where(take[labels,None],trial,x);accepted|=take
                if bool(accepted.all()):break
            damping=torch.where(accepted,damping*.5,damping*4).clamp(1e-8,1e8)
            if iteration%20==0:history.append(dict(iteration=iteration,cost=float(cost.sum()),speed_ratio=float((abs(x[1:]-x[:-1])*edge).div(caps).max()) if len(x)>1 else 0.))
        q[:,7+np.array(js)]=x
        if not torch.isfinite(q).all():raise FloatingPointError('Nonfinite sequence candidate')
        return q.cpu().numpy(),history

    @torch.inference_mode()
    def resample_wrists(self,q_np,side,targets,native_fps,output_fps=50.,limit_deg=.15,iterations=100,starts=(0,)):
        from scipy.spatial.transform import Slerp
        if len(q_np)<2:return q_np
        all_a=[];all_alpha=[];all_samples=[];all_goals=[]
        for begin,end in zip(starts,list(starts[1:])+[len(q_np)]):
            clip=q_np[begin:end]
            if len(clip)<2:continue
            oldt=np.arange(len(clip))/native_fps
            times=np.arange(int(np.floor(oldt[-1]*output_fps+1e-8))+1)/output_fps
            index=times*native_fps;a=np.floor(index+1e-9).astype(int);alpha=np.maximum(index-a,0)
            mask=(alpha>1e-8)&(a+1<len(clip));a=a[mask];alpha=alpha[mask];times=times[mask]
            if not len(a):continue
            if len(set(np.r_[a,a+1]))!=2*len(a):raise ValueError('Overlapping interpolation pairs require a coupled solver')
            sample=clip[a]*(1-alpha[:,None])+clip[a+1]*alpha[:,None]
            sample[:,3:7]=Slerp(oldt,R.from_quat(clip[:,[4,5,6,3]]))(times).as_quat()[:,[3,0,1,2]]
            all_a.append(a+begin);all_alpha.append(alpha);all_samples.append(sample);all_goals.append(Slerp(oldt,targets[begin:end])(times).as_matrix())
        if not all_a:return q_np
        a=np.concatenate(all_a);alpha=np.concatenate(all_alpha);sample=np.concatenate(all_samples)
        js=[i for i,n in enumerate(self.names) if n.startswith(side.lower()+'_wrist_')]
        body=self.m.body(side.lower()+'_wrist_yaw_link').id
        stacked=np.concatenate([q_np[a],q_np[a+1],sample]);poses=self.t(stacked)
        root_r=self.t(R.from_quat(stacked[:,[4,5,6,3]]).as_matrix())
        target_r=self.t(np.concatenate([targets[a].as_matrix(),targets[a+1].as_matrix(),np.concatenate(all_goals)]))
        n=len(a);mix=self.t(alpha)[:,None];initial=torch.cat([poses[:n,7+np.array(js)],poses[n:2*n,7+np.array(js)]],1)
        eye=torch.eye(6,device=self.device,dtype=poses.dtype).expand(n,-1,-1);threshold=np.deg2rad(limit_deg*.95)
        def residual(x):
            angles=poses[:,7:].clone();angles[:,js]=torch.cat([x[:,:3],x[:,3:],x[:,:3]*(1-mix)+x[:,3:]*mix])
            p,r,axes,pivots=self.fk(poses[:,:3],root_r,angles);_,jr=self.body_jac(p,axes,pivots,[body],js)
            e,j=self.orientation(r[:,body],target_r,jr[:,0]);z=torch.zeros_like(j[:n])
            jj=torch.cat([torch.cat([j[:n],z],2),torch.cat([z,j[n:2*n]],2),torch.cat([j[2*n:]*(1-mix[:,:,None]),j[2*n:]*mix[:,:,None]],2)])
            norm=torch.linalg.vector_norm(e,dim=1);factor=(1-threshold/norm.clamp(min=1e-12)).clamp(min=0)
            deriv=self.eye*factor[:,None,None]+threshold*(e[:,:,None]*e[:,None,:])/norm.clamp(min=1e-12)[:,None,None]**3
            deriv=torch.where((norm>threshold)[:,None,None],deriv,torch.zeros_like(deriv))
            res=torch.cat([e[:n],e[n:2*n],e[2*n:]*2,(e*factor[:,None]).reshape(3,n,3).permute(1,0,2).flatten(1)*100,(x-initial)*.001],1)
            jac=torch.cat([jj[:n],jj[n:2*n],jj[2*n:]*2,(deriv@jj).reshape(3,n,3,6).permute(1,0,2,3).flatten(1,2)*100,eye*.001],1)
            return res,jac
        # Share each edge's remaining speed budget between its two endpoints.
        # This also covers neighbouring interpolation pairs solved in parallel.
        native=self.t(q_np[:,7+np.array(js)]);caps=self.speeds[js]/native_fps
        delta=native[1:]-native[:-1];up=torch.full_like(native,float('inf'));down=up.clone()
        for begin,end in zip(starts,list(starts[1:])+[len(q_np)]):
            if end-begin<2:continue
            d=delta[begin:end-1]
            up[begin+1:end]=torch.minimum(up[begin+1:end],(caps-d).clamp(min=0)*.5)
            down[begin+1:end]=torch.minimum(down[begin+1:end],(caps+d).clamp(min=0)*.5)
            up[begin:end-1]=torch.minimum(up[begin:end-1],(caps+d).clamp(min=0)*.5)
            down[begin:end-1]=torch.minimum(down[begin:end-1],(caps-d).clamp(min=0)*.5)
        lower=torch.maximum(native-down,self.lo[js]);upper=torch.minimum(native+up,self.hi[js])
        lo=torch.cat([lower[a],lower[a+1]],1);hi=torch.cat([upper[a],upper[a+1]],1)
        x=self.lm(initial,lo,hi,residual,iterations)
        answer=q_np.copy();answer[np.ix_(a,7+np.array(js))]=x[:,:3].cpu().numpy();answer[np.ix_(a+1,7+np.array(js))]=x[:,3:].cpu().numpy()
        return answer

    @torch.inference_mode()
    def causal_arm(self,q_np,side,directions,wrist_rotation,fps,canonical,starts=(0,),iterations=40,reverse=False):
        """Causal, speed-bounded arm IK; CUDA graph replay batches whole motions."""
        ends=list(starts[1:])+[len(q_np)];lengths=[b-a for a,b in zip(starts,ends)];count=len(lengths);maxlen=max(lengths)
        js=[i for i,n in enumerate(self.names) if n.startswith(side.lower()+'_') and any(k in n for k in ('shoulder','elbow','wrist'))]
        poses=np.empty((count,maxlen,38));dirs=np.empty((count,maxlen,2,3));wr=np.empty((count,maxlen,3,3))
        for i,(a,b) in enumerate(zip(starts,ends)):
            pose=q_np[a:b][::-1] if reverse else q_np[a:b];direction=directions[a:b][::-1] if reverse else directions[a:b];wrist=wrist_rotation[a:b][::-1] if reverse else wrist_rotation[a:b]
            poses[i,:b-a]=pose;poses[i,b-a:]=pose[-1];dirs[i,:b-a]=direction;dirs[i,b-a:]=direction[-1];wr[i,:b-a]=wrist;wr[i,b-a:]=wrist[-1]
        pq=self.t(poses);dr=self.t(dirs);wrist=self.t(wr);rot=self.t(R.from_quat(poses[:,:,[4,5,6,3]].reshape(-1,4)).as_matrix().reshape(count,maxlen,3,3))
        slot=pq[:,0].clone();rs=rot[:,0].clone();ds=dr[:,0].clone();ws=wrist[:,0].clone();previous=self.t(canonical[7+np.array(js)]).expand(count,-1).clone();anchor=previous.clone()
        eye=torch.eye(7,device=self.device,dtype=pq.dtype).expand(count,-1,-1)
        def evaluate(x):
            residual,jac=self.arm_geometry(slot,rs,side,x,ds,ws)
            return torch.cat([residual,(x-anchor)*.03],1),torch.cat([jac,eye*.03],1)
        lo=self.lo[js];hi=self.hi[js]
        best=self.reflective(previous,lo,hi,evaluate,140);cost=evaluate(best)[0].square().sum(1)
        seeds=[pq[:,0,7+np.array(js)],self.t([-1.5,0,0,1.5,0,0,0]).expand(count,-1),self.t([0,1.2 if side=='Left' else -1.2,0,1.5,0,0,0]).expand(count,-1)]
        seeds.extend(self.t(x).expand(count,-1) for x in np.random.default_rng(17).uniform(lo.cpu().numpy(),hi.cpu().numpy(),(3,7)))
        for seed in seeds:
            trial=self.reflective(seed,lo,hi,evaluate,140);score=evaluate(trial)[0].square().sum(1);take=(score<cost)&(cost>=.002**2);best=torch.where(take[:,None],trial,best);cost=torch.where(take,score,cost)
        previous.copy_(best);output=torch.empty((count,maxlen,7),device=self.device,dtype=pq.dtype);output[:,0]=best
        if maxlen>1:
            counter=torch.ones(1,device=self.device,dtype=torch.long);speed=torch.minimum(self.speeds[js],self.t(28.))/fps
            def step():
                slot.copy_(pq.index_select(1,counter).squeeze(1));rs.copy_(rot.index_select(1,counter).squeeze(1));ds.copy_(dr.index_select(1,counter).squeeze(1));ws.copy_(wrist.index_select(1,counter).squeeze(1));anchor.copy_(previous)
                lower=torch.maximum(lo,previous-speed);upper=torch.minimum(hi,previous+speed)
                x=self.reflective(previous,lower,upper,evaluate,iterations);previous.copy_(x);output.index_copy_(1,counter,x[:,None]);counter.add_(1)
            # Warm up kernels on a side stream before capture. Reset all mutable state.
            stream=torch.cuda.Stream(device=self.device);stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                step()
            torch.cuda.current_stream().wait_stream(stream);previous.copy_(best);counter.fill_(1)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):step()
            previous.copy_(best);counter.fill_(1)
            for frame in range(1,maxlen):
                graph.replay()
                if frame%128==0 and getattr(self,'progress',None):
                    torch.cuda.synchronize();self.progress(dict(side=side,reverse=reverse,frame=frame,total_frames=maxlen,batch_motions=count))
            torch.cuda.synchronize()
        values=output.cpu().numpy();answer=q_np.copy()
        for i,(a,b) in enumerate(zip(starts,ends)):answer[a:b,7+np.array(js)]=values[i,:b-a][::-1] if reverse else values[i,:b-a]
        if not np.isfinite(answer).all():raise FloatingPointError('Nonfinite causal candidate')
        return answer

    @torch.inference_mode()
    def reflective(self,x,lo,hi,evaluate,iterations=80):
        """Bound-scaled damped Gauss-Newton with feasible reflected candidates."""
        x=x.clamp(lo+1e-10,hi-1e-10);eye=torch.eye(x.shape[1],device=x.device,dtype=x.dtype);damping=torch.full((len(x),),1e-8,device=x.device,dtype=x.dtype)
        def bound_stride(x,p):
            ratios=torch.where(p>0,(hi-x)/p.clamp(min=1e-30),torch.where(p<0,(lo-x)/p.clamp(max=-1e-30),torch.full_like(p,float('inf'))))
            return ratios.min(1).values,ratios
        def quad(v,H,g):return .5*(v*(H@v[:,:,None]).squeeze(-1)).sum(1)+(g*v).sum(1)
        for _ in range(iterations):
            res,jac=evaluate(x);jt=jac.transpose(-1,-2);gradient=(jt@res[:,:,None]).squeeze(-1)
            v=torch.where(gradient<0,hi-x,torch.where(gradient>0,x-lo,torch.ones_like(x))).clamp(min=1e-15)
            d=torch.sqrt(v);diag=gradient*torch.sign(gradient);gh=gradient*d
            H=(jt@jac)*d[:,:,None]*d[:,None,:]+torch.diag_embed(diag)
            ph=torch.linalg.solve_ex(H+eye*(damping[:,None,None]+1e-12),-gh[:,:,None],check_errors=False).result.squeeze(-1)
            p=d*ph;inside=((x+p>=lo)&(x+p<=hi)).all(1)
            stride,ratios=bound_stride(x,p);stride=stride.clamp(max=1.)
            hit=ratios<=stride[:,None]+1e-12;rh=torch.where(hit,-ph,ph);pp=ph*stride[:,None]
            theta=torch.maximum(torch.full_like(stride,.995),1-(gradient*v).abs().max(1).values)
            feasible=pp*theta[:,None];fscore=quad(feasible,H,gh)
            upper,_=bound_stride(x+d*pp,d*rh);upper=(upper*theta).clamp(min=0,max=1e6)
            aa=.5*(rh*(H@rh[:,:,None]).squeeze(-1)).sum(1);bb=(rh*((H@pp[:,:,None]).squeeze(-1)+gh)).sum(1)
            scalar=(-bb/(2*aa).clamp(min=1e-30)).clamp(min=0);scalar=torch.minimum(scalar,upper)
            reflected=pp+scalar[:,None]*rh;rscore=quad(reflected,H,gh)
            ag=-gh;upper,_=bound_stride(x,d*ag);upper=(upper*theta).clamp(min=0,max=1e6)
            aa=.5*(ag*(H@ag[:,:,None]).squeeze(-1)).sum(1);bb=(gh*ag).sum(1);scalar=torch.minimum((-bb/(2*aa).clamp(min=1e-30)).clamp(min=0),upper)
            steepest=ag*scalar[:,None];sscore=quad(steepest,H,gh)
            take_r=rscore<fscore;step=torch.where(take_r[:,None],reflected,feasible);score=torch.minimum(rscore,fscore)
            step=torch.where((sscore<score)[:,None],steepest,step);step=torch.where(inside[:,None],ph,step)
            trial=(x+d*step).clamp(lo+1e-12,hi-1e-12);rr,_=evaluate(trial)
            old=res.square().sum(1);new=rr.square().sum(1);take=new<old
            x=torch.where(take[:,None],trial,x);damping=torch.where(take,damping*.25,damping*10).clamp(1e-12,1e10)
        return x

    @torch.inference_mode()
    def branch_arm(self,q_np,side,directions,wrist_rotation,fps,canonical,starts=(0,),iterations=100):
        """Choose a temporally coherent path through GPU IK branches on CUDA."""
        js=[i for i,n in enumerate(self.names) if n.startswith(side.lower()+'_') and any(k in n for k in ('shoulder','elbow','wrist'))]
        ends=list(starts[1:])+[len(q_np)];lengths=[b-a for a,b in zip(starts,ends)];B=len(starts);T=max(lengths)
        q=self.t(q_np);root=self.t(R.from_quat(q_np[:,[4,5,6,3]]).as_matrix());dirs=self.t(directions);wr=self.t(wrist_rotation)
        seeds=[q[:,7+np.array(js)],self.t(canonical[7+np.array(js)]).expand(len(q),-1),self.t([-1.5,0,0,1.5,0,0,0]).expand(len(q),-1),self.t([0,1.2 if side=='Left' else -1.2,0,1.5,0,0,0]).expand(len(q),-1)]
        seeds.extend(self.t(x).expand(len(q),-1) for x in np.random.default_rng(17).uniform(self.lo[js].cpu().numpy(),self.hi[js].cpu().numpy(),(3,7)))
        # Shoulder yaw and wrist roll can compensate each other. Seed both
        # bounded wrist branches so temporal selection can avoid a later wrap.
        for yaw in (-1.5,1.5):
            for roll in (-2.5,2.5):
                seed=q[:,7+np.array(js)].clone();seed[:,2]=yaw;seed[:,4]=roll;seeds.append(seed)
        K=len(seeds);initial=torch.stack(seeds,1).reshape(-1,7);qq=q.repeat_interleave(K,0);rr=root.repeat_interleave(K,0);dd=dirs.repeat_interleave(K,0);ww=wr.repeat_interleave(K,0)
        def evaluate(x):return self.arm_geometry(qq,rr,side,x,dd,ww)
        values=self.reflective(initial,self.lo[js],self.hi[js],evaluate,iterations);geometry=evaluate(values)[0].square().sum(1).reshape(len(q),K);values=values.reshape(len(q),K,7)
        states=torch.empty((B,T,K,7),device=self.device,dtype=q.dtype);costs=torch.zeros((B,T,K),device=self.device,dtype=q.dtype)
        for i,(a,b) in enumerate(zip(starts,ends)):
            states[i,:b-a]=values[a:b];states[i,b-a:]=values[b-1];costs[i,:b-a]=geometry[a:b]
        dp=costs[:,0]+.03**2*(states[:,0]-self.t(canonical[7+np.array(js)])).square().sum(-1)
        pointers=torch.empty((B,T,K),device=self.device,dtype=torch.long);pointers[:,0]=0
        lens=torch.tensor(lengths,device=self.device);identity=torch.arange(K,device=self.device).expand(B,-1);caps=self.speeds[js]*.98/fps
        for frame in range(1,T):
            delta=states[:,frame,None]-states[:,frame-1,:,None];over=(abs(delta)-caps).clamp(min=0)
            transitions=dp[:,:,None]+over.square().sum(-1)*1500**2+delta.square().sum(-1)*.01**2
            next_cost,parent=transitions.min(1);active=frame<lens;dp=torch.where(active[:,None],next_cost+costs[:,frame],dp);pointers[:,frame]=torch.where(active[:,None],parent,identity)
        best=dp.argmin(1);batch=torch.arange(B,device=self.device);selected=torch.empty((B,T,7),device=self.device,dtype=q.dtype)
        for frame in range(T-1,-1,-1):selected[:,frame]=states[batch,frame,best];best=pointers[batch,frame,best]
        result=q_np.copy();selected=selected.cpu().numpy()
        for i,(a,b) in enumerate(zip(starts,ends)):result[a:b,7+np.array(js)]=selected[i,:b-a]
        return result
