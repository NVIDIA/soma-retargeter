"""Batched CUDA pelvis/waist IK, with analytic MJCF Jacobians.

Same foot/torso targets, bounds and residual weights as Solver.body. Frames
are solved in parallel; the weak causal previous-frame prior uses the previous
iterate (Jacobi), instead of CPU's completed previous-frame solution.
CPU MuJoCo validation remains independent of this solver.
"""
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


def skew(v):
    z=torch.zeros_like(v[...,0]);x,y,w=v.unbind(-1)
    return torch.stack([z,-w,y,w,z,-x,-y,x,z],-1).reshape(v.shape[:-1]+(3,3))


def log_rotation(m):
    # Stable matrix->quaternion conversion, including rotations near pi.
    m00,m01,m02=m[...,0,0],m[...,0,1],m[...,0,2]
    m10,m11,m12=m[...,1,0],m[...,1,1],m[...,1,2]
    m20,m21,m22=m[...,2,0],m[...,2,1],m[...,2,2]
    qa=torch.sqrt(torch.clamp(torch.stack([1+m00+m11+m22,1+m00-m11-m22,1-m00+m11-m22,1-m00-m11+m22],-1),min=0))
    candidates=torch.stack([torch.stack([qa[...,0]**2,m21-m12,m02-m20,m10-m01],-1),
                            torch.stack([m21-m12,qa[...,1]**2,m10+m01,m02+m20],-1),
                            torch.stack([m02-m20,m10+m01,qa[...,2]**2,m12+m21],-1),
                            torch.stack([m10-m01,m20+m02,m21+m12,qa[...,3]**2],-1)],-2)
    candidates=candidates/(2*qa.clamp(min=.1)[...,None])
    idx=qa.argmax(-1);q=torch.gather(candidates,-2,idx[...,None,None].expand(idx.shape+(1,4))).squeeze(-2)
    q=torch.where(q[...,:1]<0,-q,q)
    n=torch.linalg.vector_norm(q[...,1:],dim=-1,keepdim=True)
    scale=2*torch.atan2(n,q[...,:1].clamp(min=1e-15))/n.clamp(min=1e-15)
    return q[...,1:]*torch.where(n<1e-8,torch.full_like(scale,2),scale)


class BatchedWaistIK:
    def __init__(self,sol,device='cuda:0',dtype=torch.float64):
        self.device=device;self.dtype=dtype;self.sol=sol
        self.indices=np.r_[sol.legs,sol.waist];self.targets=sol.feet+[sol.torso]
        m=sol.m;needed=set(self.targets)
        for b in list(needed):
            while b>1:b=int(m.body_parentid[b]);needed.add(b)
        self.nodes=sorted(needed-{1});self.info={};self.tensor=lambda a:torch.as_tensor(a,dtype=dtype,device=device)
        self.eye=self.tensor(np.eye(3));self.eye18=self.tensor(np.eye(18))
        self.qcolumn={int(m.jnt_bodyid[j+1]):k for k,j in enumerate(self.indices)}
        masks=[]
        for target in self.targets:
            ancestors=[];b=target
            while b>1:ancestors.append(b);b=int(m.body_parentid[b])
            masks.append([int(m.jnt_bodyid[j+1]) in ancestors for j in self.indices])
        self.masks=self.tensor(masks)
        for b in self.nodes:
            j=int(m.body_jntadr[b]);assert m.body_jntnum[b]==1 and b in self.qcolumn
            self.info[b]=(int(m.body_parentid[b]),self.tensor(m.body_pos[b]),self.tensor(R.from_quat(m.body_quat[b][[1,2,3,0]]).as_matrix()),self.tensor(m.jnt_pos[j]),self.tensor(m.jnt_axis[j]),float(m.qpos0[m.jnt_qposadr[j]]),self.qcolumn[b])
        self.lo=self.tensor(sol.lo[self.indices]);self.hi=self.tensor(sol.hi[self.indices])

    def fk(self,x,root_rotation):
        positions={1:x[:,:3]};rotations={1:root_rotation};axes=[None]*15;pivots=[None]*15
        for b in self.nodes:
            parent,bpos,base,jpos,axis,ref,col=self.info[b]
            p,r=positions[parent],rotations[parent];rb=r@base
            a=x[:,3+col]-ref;K=skew(axis)
            hinge=self.eye+torch.sin(a)[:,None,None]*K+(1-torch.cos(a))[:,None,None]*(K@K)
            pivot=p+(r@bpos[:,None]).squeeze(-1)+(rb@jpos[:,None]).squeeze(-1)
            axes[col]=(rb@axis[:,None]).squeeze(-1);pivots[col]=pivot
            positions[b]=pivot-(rb@hinge@jpos[:,None]).squeeze(-1)
            rotations[b]=rb@hinge
        return torch.stack([positions[b] for b in self.targets],1),torch.stack([rotations[b] for b in self.targets],1),torch.stack(axes,1),torch.stack(pivots,1)

    def residual(self,x,root_rotation,target_p,target_r,original,prior,prior_mask,jacobian=True):
        p,r,axes,pivots=self.fk(x,root_rotation)
        rotation_error=log_rotation(target_r.transpose(-1,-2)@r)
        res=torch.cat([(p[:,:2]-target_p).reshape(len(x),6)*150,rotation_error[:,:2].reshape(len(x),6)*15,rotation_error[:,2]*8,(x-original)*.06,(x-prior)*prior_mask[:,None]*.04],1)
        if not jacobian:return res
        axis=axes[:,None]*self.masks[None,:,:,None]
        jp=torch.cross(axis,p[:,:,None]-pivots[:,None],dim=-1).transpose(-1,-2)
        jp=torch.cat([self.eye.expand(len(x),3,3,3),jp],-1)
        v=rotation_error;K=skew(v);theta=torch.linalg.vector_norm(v,dim=-1)
        coefficient=(1-.5*theta/torch.tan(.5*theta).clamp(min=1e-15))/theta.square().clamp(min=1e-15)
        coefficient=torch.where(theta<1e-4,torch.full_like(coefficient,1/12),coefficient)
        inv_left=self.eye-.5*K+coefficient[...,None,None]*(K@K)
        jr=inv_left@target_r.transpose(-1,-2)@axis.transpose(-1,-2)
        jr=torch.cat([torch.zeros_like(jr[...,:3]),jr],-1)
        J=torch.cat([jp[:,:2].reshape(len(x),6,18)*150,jr[:,:2].reshape(len(x),6,18)*15,jr[:,2]*8,self.eye18.expand(len(x),18,18)*.06,self.eye18.expand(len(x),18,18)*prior_mask[:,None,None]*.04],1)
        return res,J

    @torch.inference_mode()
    def solve(self,root,q,target_quat,target_feet,target_foot_rot,target_torso,starts,iterations=60,frame_chunk=8192):
        original_np=np.concatenate([root,q[:,self.indices]],1);answer=np.empty_like(original_np)
        starts=set(starts);info=[]
        for base in range(0,len(q),frame_chunk):
            stop=min(base+frame_chunk,len(q));original=self.tensor(original_np[base:stop]);x=original.clone()
            low=torch.cat([original[:,:3]-self.tensor([.2,.2,.18]),self.lo.expand(len(x),15)],1)
            high=torch.cat([original[:,:3]+self.tensor([.2,.2,.18]),self.hi.expand(len(x),15)],1)
            x=x.clamp(low,high)
            root_r=self.tensor(R.from_quat(target_quat[base:stop]).as_matrix())
            tp=self.tensor(target_feet[base:stop]);tr=self.tensor(np.concatenate([target_foot_rot[base:stop],target_torso[base:stop,None]],1))
            mask=self.tensor([0 if i in starts else 1 for i in range(base,stop)])
            edge=self.tensor(answer[base-1] if base and base not in starts else original_np[base])
            damping=torch.full((len(x),),.01,dtype=self.dtype,device=self.device)
            step_limit=self.tensor([.04,.04,.04]+[.3]*15)
            for iteration in range(iterations):
                prior=torch.cat([edge[None],x[:-1]],0)
                residual,J=self.residual(x,root_r,tp,tr,original,prior,mask)
                jt=J.transpose(-1,-2);normal=jt@J;rhs=(jt@residual[:,:,None]).squeeze(-1)
                step=torch.linalg.solve(normal+damping[:,None,None]*self.eye18,-rhs).clamp(-step_limit,step_limit)
                trial=(x+step).clamp(low,high)
                trial_res=self.residual(trial,root_r,tp,tr,original,prior,mask,False)
                accept=trial_res.square().sum(1)<=residual.square().sum(1)+1e-15
                x=torch.where(accept[:,None],trial,x)
                damping=torch.where(accept,damping*.5,damping*4).clamp(1e-8,1e8)
            answer[base:stop]=x.cpu().numpy()
            info.append(dict(first_frame=base,frames=stop-base,iterations=iterations))
        corrected=q.copy();corrected[:,self.indices]=answer[:,3:]
        return answer[:,:3],corrected,info
