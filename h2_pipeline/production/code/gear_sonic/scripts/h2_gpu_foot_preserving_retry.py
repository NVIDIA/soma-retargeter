"""Time-coupled lower-body IK candidate, with independent trajectory admission.

Root rotation and every upper-body joint are fixed. Root translation and legs
are optimized jointly against the existing foot trajectory and BVH targets.
Finite penalties do NOT certify feasibility: the caller must check both native
and training-rate feet, all original audit metrics, joint limits and speeds.
"""
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R,Slerp


@torch.inference_mode()
def solve_lower(solver, q_np, targets_np, names, fps, starts, iterations=160,
                root_radius=.10, progress=None):
    s=solver;q=s.t(q_np);root_r=s.t(R.from_quat(q_np[:,[4,5,6,3]]).as_matrix())
    js=[i for i,n in enumerate(s.names) if any(k in n for k in ('hip','knee','ankle'))]
    bodies=[s.m.body(n+'_ankle_pitch_link').id for n in ('left','right')]
    needed=set()
    for body in bodies:
        while body>1:needed.add(body);body=int(s.m.body_parentid[body])
    nodes=[row for row in s.nodes if row[0] in needed]
    p0,r0,_,_=s.fk(q[:,:3],root_r,q[:,7:],nodes)
    foot_p=p0[:,bodies].clone();foot_r=r0[:,bodies].clone()
    # The training loader interpolates joint angles, so native-rate foot
    # constraints alone do not preserve the training-rate FK trajectory.
    sample_ids=[];fractions=[];sample_rotations=[]
    for begin,end in zip(starts,list(starts[1:])+[len(q)]):
        old=np.arange(end-begin)/fps
        times=np.arange(int(np.floor(old[-1]*50.+1e-8))+1)/50.
        index=times*fps;left=np.floor(index+1e-9).astype(int);alpha=np.maximum(index-left,0.)
        keep=(alpha>1e-8)&(left+1<end-begin)
        sample_ids.extend((left[keep]+begin).tolist());fractions.extend(alpha[keep].tolist())
        if keep.any():sample_rotations.append(Slerp(old,R.from_quat(q_np[begin:end,[4,5,6,3]]))(times[keep]).as_matrix())
    ids=torch.tensor(sample_ids,device=s.device,dtype=torch.long)
    mix=s.t(fractions)[:,None]
    sample_q=q[ids]*(1-mix)+q[ids+1]*mix
    sample_root_r=s.t(np.concatenate(sample_rotations)) if sample_rotations else root_r[:0]
    sample_p,sample_r,_,_=s.fk(sample_q[:,:3],sample_root_r,sample_q[:,7:],nodes)
    sample_goal_p=sample_p[:,bodies].clone();sample_goal_r=sample_r[:,bodies].clone()
    target_ids=[names.index(side+'Foot') for side in ('Left','Right')]
    target_p=s.t(targets_np[:,target_ids,:3])
    target_r=s.t(R.from_quat(targets_np[:,target_ids,3:].reshape(-1,4)).as_matrix().reshape(len(q),2,3,3))
    anchor=torch.cat([q[:,:3],q[:,7+np.array(js)]],1);x=anchor.clone();dim=x.shape[1]
    lo=torch.cat([anchor[:,:3]-root_radius,s.lo[js].expand(len(q),-1)],1)
    hi=torch.cat([anchor[:,:3]+root_radius,s.hi[js].expand(len(q),-1)],1)
    eye=torch.eye(dim,device=s.device,dtype=x.dtype)
    edge=torch.ones((len(x)-1,1),device=s.device,dtype=x.dtype)
    for start in starts:
        if start:edge[start-1]=0
    triple=edge[:-1]*edge[1:]
    labels=torch.empty(len(x),device=s.device,dtype=torch.long)
    for label,(begin,end) in enumerate(zip(starts,list(starts[1:])+[len(x)])):labels[begin:end]=label
    def sums(values,idx=labels):
        return torch.zeros(len(starts),device=s.device,dtype=x.dtype).index_add_(0,idx,values)
    caps=torch.cat([torch.full((3,),float('inf'),device=s.device,dtype=x.dtype),s.speeds[js]*.98/fps])
    anchor_weight=s.t([10.]*3+[.02]*len(js));smooth_weight=s.t([30.]*3+[.2]*len(js))
    velocity_weight=10000.
    def temporal(v):
        d=(v[1:]-v[:-1])*edge
        over=(d.abs()-caps).clamp(min=0)*torch.sign(d)
        second=(v[2:]-2*v[1:-1]+v[:-2])*triple
        return d,over,second
    def transpose(d1,d2):
        d1=d1*edge;d2=d2*triple;out=torch.zeros_like(x)
        out[:-1]-=d1;out[1:]+=d1
        out[:-2]+=d2;out[1:-1]-=2*d2;out[2:]+=d2
        return out
    def objective(v):
        angles=q[:,7:].clone();angles[:,js]=v[:,3:]
        pos,rot,axes,pivots=s.fk(v[:,:3],root_r,angles,nodes)
        jp,jr=s.body_jac(pos,axes,pivots,bodies,js)
        jp=torch.cat([s.eye.expand(len(x),2,3,3),jp],-1)
        jr=torch.cat([torch.zeros((len(x),2,3,3),device=s.device,dtype=x.dtype),jr],-1)
        er,jr_anchor=s.orientation(rot[:,bodies],foot_r,jr)
        eb,jr_bvh=s.orientation(rot[:,bodies],target_r,jr)
        residual=torch.cat([(pos[:,bodies]-foot_p).flatten(1)*100000.,er.flatten(1)*10000.,
            (pos[:,bodies]-target_p).flatten(1)*20.,eb.flatten(1)*20.,(v-anchor)*anchor_weight],1)
        jac=torch.cat([jp.flatten(1,2)*100000.,jr_anchor.flatten(1,2)*10000.,
            jp.flatten(1,2)*20.,jr_bvh.flatten(1,2)*20.,(eye*anchor_weight[:,None]).expand(len(x),-1,-1)],1)
        sample_v=v[ids]*(1-mix)+v[ids+1]*mix
        sample_angles=sample_q[:,7:].clone();sample_angles[:,js]=sample_v[:,3:]
        sp,sr,sa,sj=s.fk(sample_v[:,:3],sample_root_r,sample_angles,nodes)
        sjp,sjr=s.body_jac(sp,sa,sj,bodies,js)
        sjp=torch.cat([s.eye.expand(len(ids),2,3,3),sjp],-1)
        sjr=torch.cat([torch.zeros((len(ids),2,3,3),device=s.device,dtype=x.dtype),sjr],-1)
        se,sje=s.orientation(sr[:,bodies],sample_goal_r,sjr)
        sample_res=torch.cat([(sp[:,bodies]-sample_goal_p).flatten(1)*100000.,se.flatten(1)*10000.],1)
        sample_jac=torch.cat([sjp.flatten(1,2)*100000.,sje.flatten(1,2)*10000.],1)
        diff,over,second=temporal(v)
        cost=sums(residual.square().sum(1))+sums((over*velocity_weight).square().sum(1),labels[:-1])+sums((second*smooth_weight).square().sum(1),labels[:-2])+sums(sample_res.square().sum(1),labels[ids])
        return cost,residual,jac,diff,over,second,sample_res,sample_jac
    damping=torch.full((len(starts),),.001,device=s.device,dtype=x.dtype)
    step_cap=s.t([.005]*3+[.08]*len(js))
    for iteration in range(iterations):
        cost,res,jac,diff,over,second,sres,sjac=objective(x);jt=jac.transpose(-1,-2);normal=jt@jac
        sjt=sjac.transpose(-1,-2);sample_normal=sjt@sjac
        sample_gradient=(sjt@sres[:,:,None]).squeeze(-1)
        active=(diff.abs()>caps).to(x.dtype)*velocity_weight**2
        gradient=(jt@res[:,:,None]).squeeze(-1)+transpose(over*velocity_weight**2,second*smooth_weight.square())
        gradient.index_add_(0,ids,sample_gradient*(1-mix));gradient.index_add_(0,ids+1,sample_gradient*mix)
        free=~(((x<=lo+1e-9)&(gradient>0))|((x>=hi-1e-9)&(gradient<0)))
        gradient*=free
        diagonal=normal.diagonal(dim1=-2,dim2=-1).clamp(min=1.)*damping[labels,None]+1e-6
        pre=diagonal.clone();pre[:-1]+=active;pre[1:]+=active
        pre[:-2]+=triple*smooth_weight.square();pre[1:-1]+=4*triple*smooth_weight.square();pre[2:]+=triple*smooth_weight.square()
        pre_normal=normal.clone()
        pre_normal.index_add_(0,ids,sample_normal*(1-mix[:,:,None]).square())
        pre_normal.index_add_(0,ids+1,sample_normal*mix[:,:,None].square())
        chol=torch.linalg.cholesky(pre_normal+torch.diag_embed(pre)+eye*1e-8)
        def matvec(v):
            vv=v*free
            result=(normal@vv[:,:,None]).squeeze(-1)+diagonal*vv+transpose((vv[1:]-vv[:-1])*active,(vv[2:]-2*vv[1:-1]+vv[:-2])*smooth_weight.square())
            sample_v=vv[ids]*(1-mix)+vv[ids+1]*mix
            sample_hv=(sample_normal@sample_v[:,:,None]).squeeze(-1)
            result.index_add_(0,ids,sample_hv*(1-mix));result.index_add_(0,ids+1,sample_hv*mix)
            return result*free+v*(~free)
        def precondition(v):return torch.cholesky_solve((v*free)[:,:,None],chol).squeeze(-1)*free
        step=torch.zeros_like(x);residual=-gradient;z=precondition(residual);direction=z.clone();rz=sums((residual*z).sum(1))
        for _ in range(40):
            hd=matvec(direction);alpha=rz/sums((direction*hd).sum(1)).clamp(min=1e-30)
            step+=alpha[labels,None]*direction;residual-=alpha[labels,None]*hd
            z=precondition(residual);newrz=sums((residual*z).sum(1));beta=newrz/rz.clamp(min=1e-30)
            direction=z+beta[labels,None]*direction;rz=newrz
        step=step.clamp(-step_cap,step_cap);base=x.clone();accepted=torch.zeros(len(starts),device=s.device,dtype=torch.bool)
        for scale in (1.,.5,.25,.125,.0625):
            trial=(base+scale*step).clamp(lo,hi);newcost,*_=objective(trial)
            take=(newcost<=cost)&~accepted;x=torch.where(take[labels,None],trial,x);accepted|=take
        damping=torch.where(accepted,damping*.5,damping*4).clamp(1e-9,1e8)
        if progress and (iteration%20==0 or iteration+1==iterations):
            progress(dict(iteration=iteration+1,iterations=iterations,cost=float(cost.sum())))
    q[:,:3]=x[:,:3];q[:,7+np.array(js)]=x[:,3:]
    if not torch.isfinite(q).all():raise FloatingPointError('Nonfinite lower-body candidate')
    return q.cpu().numpy()
