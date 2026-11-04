"""Upper-body feasibility SQP. Intermediate outputs stay in /tmp.

Root, legs, native timebase and frozen G1 thresholds are immutable. Only
independently accepted candidates are copied to the existing artifact tree.
"""
from pathlib import Path
import argparse,csv,importlib.util,json,shutil,sys,tempfile,time
import xml.etree.ElementTree as ET
import numpy as np
import mujoco,osqp
from scipy import sparse
from scipy.optimize import minimize,Bounds,LinearConstraint
from scipy.spatial.transform import Rotation as R,Slerp
from native_projection import BASE,digest,save,skew,log_jac,sample,load

ROOT=BASE/'artifacts/h2_foot_trajectory_v10'
BUNDLE=BASE/'artifacts/h2_bvh_gpu_staged_v5'
SOURCE=BASE/'artifacts/h2_bvh_gpu_refine131k_116014_v6'
SCRIPTS=BASE/'workspace/GR00T-WholeBodyControl-h2-clean/gear_sonic/scripts'
KEYS=['reach_jump_R_001__A107_M','confusion_001__A118_M','confusion_002__A118',
      'drinking_bottle_throw_270_R_001__A475','dance_vouge_vouge_pose_iii_R_loop_002__A321']

def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def angle(a,b):return float(np.arccos(np.clip(a@b,-1,1)))
def angular(a,b,ja,jb):
    c=float(np.clip(a@b,-1,1));return angle(a,b),-(b@ja+a@jb)/max(np.sqrt(max(0,1-c*c)),1e-8)

class Problem:
    def __init__(self,key,job,side=None,initial=None,arms_only=False):
        self.key=key;self.job=job;self.side=side
        self.m=mujoco.MjModel.from_xml_path(str(BUNDLE/'code/gear_sonic/data/assets/robot_description/mjcf/h2.xml'));self.d=mujoco.MjData(self.m)
        self.names=[self.m.joint(i).name for i in range(1,self.m.njnt)]
        self.js=np.array([i for i,n in enumerate(self.names) if any(s in n for s in ('waist','shoulder','elbow','wrist')) and (not arms_only or 'waist' not in n) and (side is None or n.startswith(side.lower()+'_'))]);self.dofs=6+self.js;self.dim=len(self.js)
        self.seed=ROOT/'existing_eight_reaudit/refined/csv'/f'{key}.csv'
        self.header,self.raw,self.cols,self.q0=load(self.seed,self);self.n=len(self.raw);self.N=self.n*self.dim
        self.seed_report=json.loads((ROOT/'existing_eight_reaudit/reports'/f'{key}.json').read_text())
        if initial:
            hh,raw,cols,q=load(initial,self);assert hh==self.header and q.shape==self.q0.shape
            protected=[i for i in range(self.m.nq) if i<7 or not any(s in self.names[i-7] for s in ('waist','shoulder','elbow','wrist'))]
            assert np.array_equal(q[:,protected],self.q0[:,protected])
            report=json.loads((Path(initial).parents[2]/'reports'/f'{key}.json').read_text())
            assert report['refined_sha256']==digest(initial) and report['physical_and_margin_pass'] and report['trajectory_guard']['passed']
            self.raw,self.cols,self.q0,self.seed_report=raw,cols,q,report
        self.original=json.loads((SOURCE/'reports'/f'{key}.json').read_text())
        self.checks=[c for c in self.original['checks'] if c['section'] in ('arms','torso') and (side is None or (c['section']=='arms' and c.get('side')==side))]
        self.arm_bodies={s:[self.m.body(s.lower()+'_'+n+'_link').id for n in ('shoulder_roll','elbow','wrist_yaw')] for s in ('Left','Right')}
        self.torso=self.m.body('torso_link').id
        margin=np.array([np.deg2rad(2)+1e-6 if any(v in self.names[j] for v in ('shoulder','wrist')) else 0. for j in self.js])
        self.lo=self.m.jnt_range[1+self.js,0]+margin;self.hi=self.m.jnt_range[1+self.js,1]-margin
        speeds={j.get('name'):float(j.find('limit').get('velocity')) for j in ET.parse(BUNDLE/'code/gear_sonic/data/assets/robot_description/urdf/h2/h2.urdf').getroot().findall('joint') if j.find('limit') is not None}
        self.speeds=np.array([speeds[n] for n in self.names]);self.steps=self.speeds[self.js]/120.
        self.x0=self.q0[:,7+self.js].copy();self.scale=.03
        z=np.load(Path(job['work'])/'h2'/f'{key}_trace.npz');human=z['source_world'];sn=z['source_names'].tolist();self.old=np.arange(self.n)/120.
        zero=np.load(BUNDLE/'profile/reference.npz');zn=zero['source_names'].tolist();zero=zero['source'];scaler=json.loads((BUNDLE/'profile/scaler.json').read_text())
        heading=R.from_euler('z',(R.from_quat(zero[zn.index('Hips'),3:])*R.from_quat([.5,.5,.5,.5])).as_euler('xyz')[2])
        self.refs={}
        for rate in (120,50):
            tt=np.arange(int(np.floor(self.old[-1]*rate+1e-8))+1)/rate
            l=np.minimum(np.floor(tt*120+1e-9).astype(int),self.n-1);r=np.minimum(l+1,self.n-1);a=np.maximum(0,tt*120-l)
            hp=np.stack([np.stack([np.interp(tt,self.old,human[:,j,k]) for k in range(3)],1) for j in range(len(sn))],1)
            def rot(name):return Slerp(self.old,R.from_quat(human[:,sn.index(name),3:]))(tt)
            hr=rot('Hips')*R.from_quat(zero[zn.index('Hips'),3:]).inv()*heading;theta=hr.as_euler('xyz')[:,2]
            ref=dict(tt=tt,l=l,r=r,a=a,left=np.c_[-np.sin(theta),np.cos(theta),np.zeros(len(tt))],torso=(rot('Chest')*R.from_quat(zero[zn.index('Chest'),3:]).inv()*heading).as_matrix(),arms={})
            for side in ('Left','Right'):
                v=np.diff(hp[:,[sn.index(side+s) for s in ('Arm','ForeArm','Hand')]],axis=1);v/=np.linalg.norm(v,axis=-1,keepdims=True)
                flex=np.arccos(np.clip((v[:,0]*v[:,1]).sum(1),-1,1))
                ref['arms'][side]=dict(dirs=v,flex=flex,straight=flex<=np.deg2rad(15),wrist=(rot(side+'Hand')*R.from_quat(scaler['joint_offsets'][side+'Hand'][1])).as_matrix())
            self.refs[rate]=ref

    def assemble(self,x):
        q=self.q0.copy();q[:,7+self.js]=x;return q

    def instant(self,q,ref,k,jac):
        self.d.qpos[:]=q;mujoco.mj_kinematics(self.m,self.d)
        if jac:mujoco.mj_comPos(self.m,self.d)
        cache={};values={};grads={};deg=180/np.pi;zeros=np.zeros((3,self.dim))
        def body(b):
            if b not in cache:
                jp=np.zeros((3,self.m.nv));jr=jp.copy()
                if jac:mujoco.mj_jacBody(self.m,self.d,jp,jr,b)
                cache[b]=(self.d.xpos[b].copy(),self.d.xmat[b].reshape(3,3).copy(),jp[:,self.dofs],jr[:,self.dofs])
            return cache[b]
        def put(section,side,name,v,g):
            key=(section,side,name);values[key]=float(v);grads[key]=g
        def orientation(matrix,target,jr):
            e=(R.from_matrix(target).inv()*R.from_matrix(matrix)).as_rotvec();theta=np.linalg.norm(e)
            return theta*deg,(e/max(theta,1e-12))@target.T@jr*deg
        for side in ('Left','Right'):
            bodies=[body(b) for b in self.arm_bodies[side]];positions=np.array([b[0] for b in bodies]);jp=np.array([b[2] for b in bodies]);vec=np.diff(positions,axis=0);jv=np.diff(jp,axis=0);length=np.linalg.norm(vec,axis=1);u=vec/length[:,None]
            jl=np.einsum('ij,ijk->ik',u,jv);ju=np.array([((np.eye(3)-np.outer(u[i],u[i]))/length[i])@jv[i] for i in range(2)])
            hr=ref['arms'][side];human=hr['dirs'][k]
            for i,name in enumerate(('upper_direction_error_deg','forearm_direction_error_deg')):
                v,g=angular(u[i],human[i],ju[i],zeros);put('arms',side,name,v*deg,g*deg)
                for axis in range(3):put('aux',side,f'direction_{i}_{axis}',(u[i,axis]-human[i,axis])*deg,ju[i,axis]*deg)
            flex,g=angular(u[0],u[1],ju[0],ju[1]);df=flex-hr['flex'][k]
            for name in ('elbow_flex_error_deg','elbow_flex_error_when_human_straight_deg'):put('arms',side,name,abs(df)*deg,np.sign(df)*g*deg)
            put('aux',side,'flex_signed',df*deg,g*deg)
            err=vec.sum(0)-(human*length[:,None]).sum(0);je=jv.sum(0)-np.einsum('ij,ik->jk',human,jl);norm=np.linalg.norm(err);total=length.sum()
            put('arms',side,'normalized_hand_error',norm/total,(err/max(norm,1e-12)@je)/total-norm/total**2*jl.sum(0))
            value,grad=orientation(bodies[-1][1],hr['wrist'][k],bodies[-1][3]);put('arms',side,'wrist_orientation_error_deg',value,grad)
            target=hr['wrist'][k];e=(R.from_matrix(target).inv()*R.from_matrix(bodies[-1][1])).as_rotvec()
            ej=log_jac(e)@target.T@bodies[-1][3]
            for axis in range(3):put('aux',side,f'wrist_log_{axis}',e[axis]*deg,ej[axis]*deg)
        _,matrix,_,jr=body(self.torso);up=matrix[:,2];left=ref['left'][k];target=ref['torso'][k];a=up@left;b=up[2];expected=np.arctan2(target[:,2]@left,target[2,2]);e=np.arctan2(a,b)-expected;ju=-skew(up)@jr
        put('torso','','torso_lateral_error_deg',abs(e)*deg,np.sign(e)*(b*(left@ju)-a*ju[2])/max(a*a+b*b,1e-12)*deg)
        value,grad=orientation(matrix,target,jr);put('torso','','torso_orientation_error_deg',value,grad)
        e=(R.from_matrix(target).inv()*R.from_matrix(matrix)).as_rotvec();ej=log_jac(e)@target.T@jr
        for axis in range(3):put('aux','',f'torso_log_{axis}',e[axis]*deg,ej[axis]*deg)
        if hasattr(self,'fixed_goal'):
            for i,b in enumerate(self.fixed_goal['bodies']):
                pos,rotation,jp,jr=body(b);dp=(pos-ref['fixed_position'][k,i])/.0005
                for axis in range(3):put('fixed','',f'position_{i}_{axis}',dp[axis],jp[axis]/.0005)
                if i in (2,5,6):
                    target=ref['fixed_rotation'][k,i];e=(R.from_matrix(target).inv()*R.from_matrix(rotation)).as_rotvec();jlog=log_jac(e)@target.T@jr
                    for axis in range(3):put('fixed','',f'rotation_{i}_{axis}',e[axis]/np.deg2rad(.1),jlog[axis]/np.deg2rad(.1))
        return values,grads

    def source_residuals(self,metrics):
        records=[]
        for rate in self.refs:
            weight=np.sqrt(120/rate)
            if hasattr(self,'fixed_goal'):
                for i in range(7):
                    for axis in range(3):records.append((metrics[(rate,'fixed','',f'position_{i}_{axis}')],weight))
                    if i in (2,5,6):
                        for axis in range(3):records.append((metrics[(rate,'fixed','',f'rotation_{i}_{axis}')],weight))
                continue
            for side in ('Left','Right'):
                for segment in range(2):
                    for axis in range(3):records.append((metrics[(rate,'aux',side,f'direction_{segment}_{axis}')],weight))
                records.append((metrics[(rate,'aux',side,'flex_signed')],weight*.2))
                for axis in range(3):records.append((metrics[(rate,'aux',side,f'wrist_log_{axis}')],weight/5.))
            for axis in range(3):records.append((metrics[(rate,'aux','',f'torso_log_{axis}')],weight/.25))
        return records

    def anatomical_project(self,iterations=24,wrist_budget=15.,checkpoint=None):
        """Lexicographic primary fit followed by closest bounded hand rotation.

        Body/legs are frozen. Primary envelopes come from immutable canonical
        BVH errors, not from fitted wrist goals or changed G1 thresholds.
        """
        assert not any('waist' in self.names[j] for j in self.js)
        sides=[self.side] if self.side else ['Left','Right'];x=self.x0.copy();_,baseline,base_obs=self.evaluate(x,True)
        primary_names=('upper_direction_error_deg','forearm_direction_error_deg','elbow_flex_error_deg','normalized_hand_error')
        caps={}
        for rate in self.refs:
            for side in sides:
                for name in primary_names:
                    key=(rate,'arms',side,name);v=baseline[key][0];caps[key]=np.maximum(v,1e-6 if name=='normalized_hand_error' else .005)
        canonical_stats={}
        for key,(v,_,_,_,_) in baseline.items():
            if key[1]!='arms' or key[2] not in sides or key[3] not in primary_names:continue
            canonical_stats[key]={'mean':float(v.mean()),'p95':float(np.percentile(v,95)),'max':float(v.max())}
        def primary_records(metrics):
            records=[]
            for rate,ref in self.refs.items():
                weight=np.sqrt(120/rate)
                for side in sides:
                    for i in range(2):
                        for j in range(3):records.append((metrics[(rate,'aux',side,f'direction_{i}_{j}')],weight))
                    values,grads,l,r,a=metrics[(rate,'aux',side,'flex_signed')]
                    factor=np.where(ref['arms'][side]['straight'],2.,1.)
                    records.append(((values*factor,grads*factor[:,None] if grads is not None else None,l,r,a),weight))
            return records
        def wrist_records(metrics):
            return [(metrics[(rate,'aux',side,f'wrist_log_{j}')],np.sqrt(120/rate)) for rate in self.refs for side in sides for j in range(3)]
        def cost(records):return sum(float(v@v)*w*w for (v,_,_,_,_),w in records)
        def check(theta):
            physical,metrics,_=self.evaluate(theta,False)
            regressions=[]
            for key,stats in canonical_stats.items():
                v=metrics[key][0]
                for stat,value in [('mean',float(v.mean())),('p95',float(np.percentile(v,95))),('max',float(v.max()))]:
                    if value>stats[stat]+1e-6:regressions.append((key,stat,value-stats[stat]))
            wrist=max(float(metrics[(rate,'arms',side,'wrist_orientation_error_deg')][0].max()) for rate in self.refs for side in sides)
            worst=sorted(regressions,key=lambda z:z[2],reverse=True)[:3]
            return dict(physical=physical['physical'],speed_ratio=physical['speed_ratio'],primary_cost=cost(primary_records(metrics)),wrist_cost=cost(wrist_records(metrics)),wrist_max_deg=wrist,primary_regressions=len(regressions),regression_examples=worst),metrics
        initial,_=check(x);current=initial.copy();history=[];started=time.time();snapshot=None
        directions=np.array([[i,j,k] for i in (-1,0,1) for j in (-1,0,1) for k in (-1,0,1) if i or j or k],float);directions/=np.linalg.norm(directions,axis=1)[:,None]
        def block_entries(g,l,r,a):
            entries=[]
            for frame,mix in ((l,1-a),(r,a)):
                if mix<1e-10:continue
                entries.extend((int(frame*self.dim+j),float(v*self.scale*mix)) for j,v in enumerate(g) if abs(v)>1e-14)
            return entries
        def linear_constraints(theta,metrics,restoring=False):
            ar=[];ac=[];av=[];lower=[];upper=[]
            def constraint(entries,lo,hi):
                row=len(lower)
                for j,v in entries:ar.append(row);ac.append(j);av.append(v)
                lower.append(lo);upper.append(hi)
            for key,cap in caps.items():
                v,g,l,r,a=metrics[key];scale=max(float(np.percentile(baseline[key][0],95)),.001 if key[3]=='normalized_hand_error' else .1)
                for i in range(len(v)):constraint(block_entries(g[i]/scale,l[i],r[i],a[i]),-np.inf,(cap[i]-v[i])/scale)
                stats=canonical_stats[key];order=np.argsort(v);pos=.95*(len(v)-1);ilo=int(np.floor(pos));ihi=int(np.ceil(pos));f=pos-ilo
                for stat,ids,weights in [('mean',np.arange(len(v)),np.full(len(v),1/len(v))),('p95',np.array([order[ilo],order[ihi]]),np.array([1-f,f])),('max',np.array([v.argmax()]),np.array([1.]))]:
                    entries=[]
                    for i,w in zip(ids,weights):entries.extend(block_entries(g[i]*w/scale,l[i],r[i],a[i]))
                    value=float(v[ids]@weights)
                    headroom=min(stats[stat]*1e-4,1e-7 if key[3]=='normalized_hand_error' else .001) if restoring and value>stats[stat]+1e-6 else -2e-7
                    constraint(entries,-np.inf,(stats[stat]-headroom-value)/scale)
            for rate in self.refs:
                for side in sides:
                    components=[metrics[(rate,'aux',side,f'wrist_log_{j}')] for j in range(3)];_,_,l,r,a=components[0]
                    for i in range(len(l)):
                        v=np.array([c[0][i] for c in components]);g=np.array([c[1][i] for c in components])
                        for direction in directions:constraint(block_entries(direction@g/wrist_budget,l[i],r[i],a[i]),-np.inf,1-float(direction@v)/wrist_budget)
            for i in range(self.n):
                for j in range(self.dim):constraint([(i*self.dim+j,1.)],max(-2.,(self.lo[j]-theta[i,j])/self.scale),min(2.,(self.hi[j]-theta[i,j])/self.scale))
            for i in range(self.n-1):
                for j in range(self.dim):
                    step=self.steps[j]*.9995;diff=theta[i+1,j]-theta[i,j];constraint([(i*self.dim+j,-self.scale/step),((i+1)*self.dim+j,self.scale/step)],(-step-diff)/step,(step-diff)/step)
            A=sparse.coo_matrix((av,(ar,ac)),shape=(len(lower),self.N)).tocsc()
            return A,np.array(lower),np.array(upper)
        def admissible(score,metrics,phase):
            return score['physical'] and not score['primary_regressions'] and score['wrist_max_deg']<=wrist_budget+1e-6 and (phase=='primary' or all(np.all(metrics[key][0]<=cap+1e-6) for key,cap in caps.items()))
        def violation(score,metrics,phase):
            value=1000*max(0,score['speed_ratio']-1.000001)**2+max(0,score['wrist_max_deg']/wrist_budget-1)**2
            for key,stats in canonical_stats.items():
                v=metrics[key][0];scale=max(stats['max'],.001 if key[3]=='normalized_hand_error' else .1)
                for stat,z in [('mean',v.mean()),('p95',np.percentile(v,95)),('max',v.max())]:value+=max(0,(float(z)-stats[stat]-1e-6)/scale)**2
                if phase=='wrist':value+=float(np.sum(np.maximum((v-caps[key]-1e-6)/scale,0)**2))
            return value
        def restore(theta,score,metrics,phase):
            rows=[]
            for repair_iteration in range(3):
                if admissible(score,metrics,phase):break
                _,full,_=self.evaluate(theta,True);A,lo,hi=linear_constraints(theta,full,True)
                solver=osqp.OSQP();solver.setup(P=sparse.eye(self.N,format='csc'),q=np.zeros(self.N),A=A,l=lo,u=hi,eps_abs=1e-7,eps_rel=1e-7,max_iter=6000,polishing=True,verbose=False);solution=solver.solve()
                if solution.x is None or not np.isfinite(solution.x).all() or solution.info.status not in ('solved','solved inaccurate','maximum iterations reached'):break
                delta=solution.x.reshape(self.n,self.dim)*self.scale;old_violation=violation(score,metrics,phase);winner=None
                for alpha in (1.,.5,.25):
                    trial=np.clip(theta+alpha*delta,self.lo,self.hi);candidate,new_metrics=check(trial);amount=violation(candidate,new_metrics,phase)
                    if admissible(candidate,new_metrics,phase) or amount<old_violation-1e-15:winner=(trial,candidate,new_metrics);break
                rows.append(dict(iteration=repair_iteration+1,status=solution.info.status,primal=float(solution.info.prim_res),accepted=winner is not None))
                if winner is None:break
                theta,score,metrics=winner
            return theta,score,metrics,rows
        for phase in ('primary',):
            for iteration in range(iterations if phase=='primary' else min(iterations,12)):
                _,metrics,_=self.evaluate(x,True);records=primary_records(metrics) if phase=='primary' else wrist_records(metrics)
                jr=[];jc=[];jv=[];res=[]
                for (values,grads,l,r,a),weight in records:
                    offset=len(res);res.extend((values*weight).tolist())
                    for i,g in enumerate(grads):
                        for col,value in block_entries(g*weight,l[i],r[i],a[i]):jr.append(offset+i);jc.append(col);jv.append(value)
                J=sparse.coo_matrix((jv,(jr,jc)),shape=(len(res),self.N)).tocsc();P=J.T@J+sparse.eye(self.N,format='csc')*.02;g=np.asarray(J.T@np.array(res)).ravel();A,lower,upper=linear_constraints(x,metrics)
                solver=osqp.OSQP();solver.setup(P=sparse.triu(P).tocsc(),q=g,A=A,l=lower,u=upper,eps_abs=1e-5,eps_rel=1e-5,max_iter=4000,polishing=True,verbose=False);solution=solver.solve();accepted=False;trials=[]
                if solution.x is not None and np.isfinite(solution.x).all() and solution.info.status in ('solved','solved inaccurate','maximum iterations reached'):
                    delta=solution.x.reshape(self.n,self.dim)*self.scale
                    for alpha in (1.,.5,.25,.125,.0625,.03125,.015625,.0078125,.00390625,.001953125):
                        trial=np.clip(x+alpha*delta,self.lo,self.hi);score,trial_metrics=check(trial);metric='primary_cost' if phase=='primary' else 'wrist_cost'
                        repairs=[]
                        if alpha>=.5 and not admissible(score,trial_metrics,phase):trial,score,trial_metrics,repairs=restore(trial,score,trial_metrics,phase)
                        trials.append(dict(alpha=alpha,repairs=repairs,**score))
                        if admissible(score,trial_metrics,phase) and score[metric]<current[metric]-1e-5:x=trial;current=score;accepted=True;break
                row=dict(phase=phase,iteration=iteration+1,accepted=accepted,score=current,qp_status=solution.info.status,qp_iterations=int(solution.info.iter),qp_primal_residual=float(solution.info.prim_res),trials=trials if not accepted else trials[-1:],seconds=time.time()-started);history.append(row);print('ANATOMICAL',self.key,self.side,json.dumps(row),flush=True)
                if checkpoint:
                    meta=dict(key=self.key,side=self.side,canonical_sha256=digest(self.seed),score=current,history=history)
                    np.savez_compressed(checkpoint,x=x,metadata=np.array(json.dumps(meta)))
                if not accepted:break
        refined,yaw=refine_wrist_yaw(self,self.assemble(x),wrist_budget)
        if yaw['success']:
            proposed=refined[:,7+self.js];score,mm=check(proposed)
            if admissible(score,mm,'primary'):x=proposed;current=score
        history.append(dict(phase='bounded_wrist_yaw',score=current,optimization=yaw))
        if checkpoint:np.savez_compressed(checkpoint,x=x,metadata=np.array(json.dumps(dict(key=self.key,side=self.side,canonical_sha256=digest(self.seed),score=current,history=history))))
        return x,dict(before=initial,after=current,history=history,wrist_budget_deg=wrist_budget,body_fixed=True,source_weights=dict(direction_deg=1.,flex_deg=1.,human_straight_flex_multiplier=2.,wrist='bounded_yaw_secondary'),source_goals='BVH arm directions and flex first; bounded wrist-yaw correction second, preserving primary arm geometry and native joint constraints.')

    def source_measure(self,x,jac=False):
        physical,metrics,_=self.evaluate(x,jac)
        cost=sum(float(v@v)*w*w for (v,_,_,_,_),w in self.source_residuals(metrics))
        added=x-self.x0
        cost+=.01*float(np.sum(np.diff(added,n=2,axis=0)**2))
        return dict(cost=cost,speed_ratio=physical['speed_ratio'],bounds=physical['bounds'],physical=physical['physical']),metrics

    def source_qp(self,x,metrics):
        jr=[];jc=[];jv=[];res=[]
        for (values,grads,l,r,a),weight in self.source_residuals(metrics):
            offset=len(res);res.extend((values*weight).tolist())
            for i,g in enumerate(grads):
                for frame,mix in ((l[i],1-a[i]),(r[i],a[i])):
                    if mix<1e-10:continue
                    row=g*self.scale*weight*mix;ids=np.flatnonzero(abs(row)>1e-14)
                    jr.extend([offset+i]*len(ids));jc.extend((frame*self.dim+ids).tolist());jv.extend(row[ids].tolist())
        J=sparse.coo_matrix((jv,(jr,jc)),shape=(len(res),self.N)).tocsc();res=np.array(res)
        ar=[];ac=[];av=[];lower=[];upper=[]
        def constraint(entries,lo,hi):
            row=len(lower)
            for j,z in entries:ar.append(row);ac.append(j);av.append(z)
            lower.append(lo);upper.append(hi)
        for i in range(self.n):
            for j in range(self.dim):constraint([(i*self.dim+j,1.)],max(-2.,(self.lo[j]-x[i,j])/self.scale),min(2.,(self.hi[j]-x[i,j])/self.scale))
        for i in range(self.n-1):
            for j in range(self.dim):
                cap=self.steps[j];d=x[i+1,j]-x[i,j];constraint([(i*self.dim+j,-self.scale/cap),((i+1)*self.dim+j,self.scale/cap)],(-cap-d)/cap,(cap-d)/cap)
        D=sparse.kron(sparse.diags([np.ones(self.n-2),-2*np.ones(self.n-2),np.ones(self.n-2)],[0,1,2],shape=(self.n-2,self.n)),sparse.eye(self.dim),format='csc')
        P=J.T@J+sparse.eye(self.N,format='csc')*.002+.01*self.scale**2*(D.T@D)
        g=np.asarray(J.T@res).ravel()+.01*self.scale*np.asarray(D.T@(D@(x-self.x0).ravel())).ravel()
        A=sparse.coo_matrix((av,(ar,ac)),shape=(len(lower),self.N)).tocsc();solver=osqp.OSQP()
        solver.setup(P=sparse.triu(P).tocsc(),q=g,A=A,l=np.array(lower),u=np.array(upper),verbose=False,eps_abs=1e-6,eps_rel=1e-6,max_iter=20000,polishing=True)
        result=solver.solve();info=dict(status=result.info.status,iterations=int(result.info.iter),primal_residual=float(result.info.prim_res))
        if result.x is None or not np.isfinite(result.x).all() or result.info.status not in ('solved','solved inaccurate','maximum iterations reached'):return None,info
        return result.x.reshape(self.n,self.dim)*self.scale,info

    def reconstruct(self,iterations):
        x=self.x0.copy();before,_=self.source_measure(x);current=before;history=[];start=time.time()
        for iteration in range(iterations):
            current,metrics=self.source_measure(x,True);delta,info=self.source_qp(x,metrics);accepted=False
            if delta is not None:
                for alpha in (1.,.5,.25,.125,.0625,.03125):
                    trial=np.clip(x+alpha*delta,self.lo,self.hi);score,_=self.source_measure(trial)
                    tolerance=1e-10 if hasattr(self,'fixed_goal') else 1e-5
                    if score['physical'] and score['cost']<current['cost']-tolerance:x=trial;current=score;accepted=True;break
            history.append(dict(iteration=iteration+1,score=current,qp=info,accepted=accepted,seconds=time.time()-start));print('SOURCE_PROJECT',self.key,json.dumps(history[-1]),flush=True)
            if not accepted:break
        info=dict(before=before,after=current,history=history,source_weights=dict(direction_deg=1.,flex_deg=5.,wrist_deg=5.,torso_deg=.25),source_goals='Immutable BVH directions, flex and mapped hand/chest rotations; feasible H2 reference is their constrained joint-space projection.')
        if hasattr(self,'fixed_goal'):info.update(fixed_reference_sha256=digest(self.fixed_goal['path']),source_weights=dict(body_position_mm=.5,wrist_torso_rotation_deg=.1),source_goals='Frozen projected H2 body poses; positions 0.5mm, wrist/torso rotations 0.1deg')
        return x,info

    def continue_branch(self,diagnosis):
        assert self.side==diagnosis['side'] and self.dim==7
        assert [self.names[j] for j in self.js]==diagnosis['joint_names']
        from scipy.optimize import least_squares
        frame=int(diagnosis['frame']);angles=np.array(diagnosis['best_angles']);curve=self.x0.copy();curve[frame]=angles;ref=self.refs[120]
        terms=[(('aux',self.side,f'direction_{i}_{j}'),1.) for i in range(2) for j in range(3)]+[(('aux',self.side,'flex_signed'),.2)]+[(('aux',self.side,f'wrist_log_{j}'),.2) for j in range(3)]
        costs=np.zeros(self.n);begin=time.time()
        for order in (range(frame-1,-1,-1),range(frame+1,self.n)):
            previous=angles.copy()
            for f in order:
                q=self.q0[f].copy()
                def data(x):
                    value=q.copy();value[7+self.js]=x;v,g=self.instant(value,ref,f,True)
                    return np.r_[np.array([v[k]*w for k,w in terms]),(x-previous)*.0001],np.vstack([np.array([g[k]*w for k,w in terms]),np.eye(self.dim)*.0001])
                opt=least_squares(lambda z:data(z)[0],np.clip(previous,self.lo+1e-8,self.hi-1e-8),jac=lambda z:data(z)[1],bounds=(self.lo,self.hi),max_nfev=60,ftol=1e-9,xtol=1e-9,gtol=1e-9)
                curve[f]=opt.x;previous=opt.x.copy();costs[f]=float(np.sum(data(opt.x)[0][:-self.dim]**2))
                if f%200==0:print('BRANCH_CONTINUATION',self.key,'frame',f,'cost',costs[f],'seconds',time.time()-begin,flush=True)
        raw_speed=float((abs(np.diff(curve,axis=0))/self.steps).max())
        # Closest complete native trajectory with strict speed headroom.
        N=self.N;I=sparse.eye(N,format='csc');E=sparse.kron(sparse.diags([-np.ones(self.n-1),np.ones(self.n-1)],[0,1],shape=(self.n-1,self.n)),sparse.eye(self.dim),format='csc');caps=np.tile(self.steps*.999,self.n-1)
        A=sparse.vstack([I,E],format='csc');lower=np.r_[np.tile(self.lo,self.n),-caps];upper=np.r_[np.tile(self.hi,self.n),caps]
        solver=osqp.OSQP();solver.setup(P=I,q=-curve.ravel(),A=A,l=lower,u=upper,eps_abs=1e-8,eps_rel=1e-8,max_iter=30000,polishing=True,verbose=False);result=solver.solve()
        assert result.info.status in ('solved','solved inaccurate'),result.info.status
        projected=result.x.reshape(self.n,self.dim);self.x0=projected.copy()
        q=self.q0.copy();q[:,7+self.js]=projected;self.q0=q
        info=dict(seed_frame=frame,side=self.side,full_pose_seed=diagnosis['best'],continuation_raw_speed_ratio=raw_speed,projected_speed_ratio=float((abs(np.diff(projected,axis=0))/self.steps).max()),continuation_cost_max=float(costs.max()),projection_max_joint_change_deg=float(np.rad2deg(abs(projected-curve)).max()),seconds=time.time()-begin)
        print('BRANCH_SPEED_PROJECTION',self.key,json.dumps(info),flush=True)
        return info

    def evaluate(self,x,jac=False):
        q=self.assemble(x);allmetrics={};observed=[]
        for rate,ref in self.refs.items():
            qs=sample(q,ref['tt']);vv={};gg={}
            for k,qi in enumerate(qs):
                v,g=self.instant(qi,ref,k,jac)
                for key,value in v.items():vv.setdefault(key,[]).append(value);gg.setdefault(key,[]).append(g[key])
            for key,values in vv.items():
                mask=ref['arms'][key[1]]['straight'] if key[2]=='elbow_flex_error_when_human_straight_deg' else np.ones(len(qs),bool)
                allmetrics[(rate,*key)]=(np.array(values)[mask],np.array(gg[key])[mask] if jac else None,ref['l'][mask],ref['r'][mask],ref['a'][mask])
        for c in self.checks:
            key=(c['fps'],c['section'],c.get('side',''),c['metric']);v=allmetrics[key][0]
            if not len(v):continue
            stat=c['stat'];value=float(v.mean() if stat=='mean' else np.percentile(v,95) if stat=='p95' else v.max());scale=max(c['g1'],.0005 if c['metric']=='normalized_hand_error' else .05)
            observed.append(dict(check=c,value=value,scale=scale,over=max(0.,(value-c['g1']-1e-6)/scale)))
        over=np.array([o['over'] for o in observed]);speed=float((abs(np.diff(q[:,7:],axis=0))*120/self.speeds).max());bounds=max(float((self.lo-x).max()),float((x-self.hi).max()),0.)
        score=dict(cost=float(over@over+10*over.max()**2),failed=int((over>0).sum()),max_relative_excess=float(over.max()),speed_ratio=speed,bounds=bounds,physical=speed<=1+1e-6 and bounds<=1e-8)
        return score,allmetrics,observed

    def qp(self,x,metrics):
        rows=[];targets=[];weights=[];hard=[]
        for c in self.checks:
            v,g,l,r,a=metrics[(c['fps'],c['section'],c.get('side',''),c['metric'])]
            if not len(v):continue
            cap=c['g1']-min(c['g1']*.001,.00001 if c['metric']=='normalized_hand_error' else .005);scale=max(c['g1'],.0005 if c['metric']=='normalized_hand_error' else .05);stat=c['stat']
            current=float(v.mean() if stat=='mean' else np.percentile(v,95) if stat=='p95' else v.max());preserve=current<=c['g1']+1e-6
            selections=[]
            if stat=='mean':selections=[(np.arange(len(v)),np.full(len(v),1/len(v)),1.)]
            elif stat=='p95':
                order=np.argsort(v);point=.95*(len(v)-1);lo=int(np.floor(point));hi=int(np.ceil(point));f=point-lo
                selections=[(np.array([order[lo],order[hi]]),np.array([1-f,f]),1.)]
                if c['metric']!='wrist_orientation_error_deg':
                    # Move the relevant distribution together; a quantile
                    # derivative alone repeatedly changes just one rank.
                    kept=order[:int(np.ceil(point))+1];ids=kept[v[kept]>.75*cap]
                    selections.extend((np.array([i]),np.array([1.]),1/max(1,int((v[kept]>cap).sum()))) for i in ids)
            else:
                ids=np.flatnonzero(v>cap*.75)
                ids=np.union1d(ids,np.argsort(v)[-8:]);worst=int(v.argmax())
                selections=[(np.array([i]),np.array([1.]),10. if i==worst else 1/max(1,int((v>cap).sum()))) for i in ids]
            for ids,w,weight in selections:
                entries=[]
                for i,wi in zip(ids,w):
                    block=g[i]*wi*self.scale/scale
                    for frame,mix in ((l[i],1-a[i]),(r[i],a[i])):
                        if mix<1e-10:continue
                        entries.extend((int(frame*self.dim+j),float(z*mix)) for j,z in enumerate(block) if abs(z)>1e-14)
                rows.append(entries);targets.append((cap-float(v[ids]@w))/scale);weights.append(weight);hard.append(preserve)
        # Norm errors have no useful directional linearization at exact fit.
        # Component cuts retain all three wrist rotation directions there.
        # These are outer cuts; the exact original angular gates still decide.
        for rate in self.refs:
            for side in ('Left','Right'):
                matches=[c for c in self.checks if c['fps']==rate and c['section']=='arms' and c.get('side')==side and c['metric']=='wrist_orientation_error_deg']
                if not matches:continue
                cap=next(c['g1'] for c in matches if c['stat']=='max');p95=next(c['g1'] for c in matches if c['stat']=='p95')
                magnitude=metrics[(rate,'arms',side,'wrist_orientation_error_deg')][0]
                sample_caps=np.full(len(magnitude),cap)
                # Fix a sufficient 95% subset while allowing the existing tail.
                count=int(np.ceil(.95*(len(magnitude)-1)))+1
                sample_caps[np.argsort(magnitude)[:count]]=p95
                components=[metrics[(rate,'aux',side,f'wrist_log_{axis}')] for axis in range(3)]
                directions=np.array([[x,y,z] for x in (-1,0,1) for y in (-1,0,1) for z in (-1,0,1) if x or y or z],float);directions/=np.linalg.norm(directions,axis=1)[:,None]
                for i in range(len(magnitude)):
                    vv=np.array([z[0][i] for z in components]);gg=np.array([z[1][i] for z in components]);_,_,l,r,a=components[0];scale=max(sample_caps[i],.05)
                    for direction in directions:
                            entries=[];block=direction@gg*self.scale/scale
                            for frame,mix in ((l[i],1-a[i]),(r[i],a[i])):
                                if mix<1e-10:continue
                                entries.extend((int(frame*self.dim+j),float(z*mix)) for j,z in enumerate(block) if abs(z)>1e-14)
                            rows.append(entries);targets.append((sample_caps[i]*.999-direction@vv)/scale);weights.append(10.);hard.append(True)
        M=len(rows);ar=[];ac=[];av=[];lower=[];upper=[]
        def add(entries,lo,hi):
            k=len(lower)
            for j,z in entries:ar.append(k);ac.append(j);av.append(z)
            lower.append(lo);upper.append(hi)
        for k,(entries,cap) in enumerate(zip(rows,targets)):add(entries+[(self.N+k,-1.)],-np.inf,cap)
        for k in range(M):add([(self.N+k,1.)],0.,0. if hard[k] else np.inf)
        for i in range(self.n):
            for j in range(self.dim):add([(i*self.dim+j,1.)],max((self.lo[j]-x[i,j])/self.scale,-1.),min((self.hi[j]-x[i,j])/self.scale,1.))
        for i in range(self.n-1):
            for j in range(self.dim):
                cap=self.steps[j];diff=x[i+1,j]-x[i,j];add([(i*self.dim+j,-self.scale/cap),((i+1)*self.dim+j,self.scale/cap)],(-cap-diff)/cap,(cap-diff)/cap)
        A=sparse.coo_matrix((av,(ar,ac)),shape=(len(lower),self.N+M)).tocsc();P=sparse.diags(np.r_[np.full(self.N,.002),np.array(weights)],format='csc');linear=np.r_[1e-6*((x-self.x0)/self.scale).ravel(),np.array(weights)*.02]
        solver=osqp.OSQP();solver.setup(P=P,q=linear,A=A,l=np.array(lower),u=np.array(upper),verbose=False,eps_abs=1e-6,eps_rel=1e-6,max_iter=20000,polishing=True)
        sol=solver.solve();info=dict(status=sol.info.status,iterations=sol.info.iter,primal_residual=sol.info.prim_res,slacks=M)
        if sol.x is None or not np.isfinite(sol.x).all() or sol.info.status not in ('solved','solved inaccurate','maximum iterations reached'):return None,info
        return sol.x[:self.N].reshape(self.n,self.dim)*self.scale,info

    def window(self,knots=None):
        """Direct nonlinear restoration on the small set of failing frames."""
        score,base,observed=self.evaluate(self.x0,True);active=set()
        for o in observed:
            if o['over']<=0:continue
            c=o['check'];v,_,l,r,_=base[(c['fps'],c['section'],c.get('side',''),c['metric'])]
            ids=np.flatnonzero(v>c['g1'])
            for idx in ids:
                padding=getattr(self,'window_padding',2)
                for f in (l[idx],r[idx]):active.update(range(max(0,int(f)-padding),min(self.n,int(f)+padding+1)))
        frames=np.array(sorted(active));assert len(frames)>0
        if knots is None:assert len(frames)<=120,('Use --knots for larger windows',len(frames))
        chunks=np.split(frames,np.flatnonzero(np.diff(frames)>1)+1)
        stride=1 if knots is None else max(1,int(np.ceil(len(frames)/max(1,knots-2*len(chunks)))))
        nodes=[np.unique(np.r_[chunk[::stride],chunk[-1]]) for chunk in chunks]
        controls=sum(len(x) for x in nodes);B=np.zeros((self.n,controls));offset=0
        for chunk,ns in zip(chunks,nodes):
            for j in range(len(ns)):B[chunk,offset+j]=np.interp(chunk,ns,np.eye(len(ns))[j])
            offset+=len(ns)
        lookup=set(frames.tolist());D=controls*self.dim;cache={};history=[];start=time.time()
        print('WINDOW_LAYOUT',self.key,'frames',len(frames),'controls',controls,'variables',D,flush=True)
        records=[]
        for c in self.checks:
            key=(c['section'],c.get('side',''),c['metric']);v,g,l,r,a=base[(c['fps'],*key)]
            if not len(v):continue
            ref=self.refs[c['fps']]
            orig=np.flatnonzero(ref['arms'][key[1]]['straight']) if key[2]=='elbow_flex_error_when_human_straight_deg' else np.arange(len(ref['tt']))
            changed=np.array([i for i in range(len(v)) if int(l[i]) in lookup or int(r[i]) in lookup],int)
            records.append((c,key,v,l,r,a,orig,changed))
        needed={rate:set() for rate in self.refs}
        for c,key,v,l,r,a,orig,changed in records:needed[c['fps']].update(orig[changed].tolist())
        def evaluate(z):
            if 'z' in cache and np.array_equal(cache['z'],z):return cache['values'],cache['jac']
            full=self.x0+B@z[:-1].reshape(controls,self.dim);q=self.assemble(full);instant={}
            for rate,ids in needed.items():
                ref=self.refs[rate];qs=sample(q,ref['tt'])
                for k in ids:instant[(rate,k)]=self.instant(qs[k],ref,k,True)
            result=[];matrix=[]
            for c,key,v,l,r,a,orig,changed in records:
                values=v.copy()
                for i in changed:values[i]=instant[(c['fps'],int(orig[i]))][0][key]
                stat=c['stat']
                if stat=='mean':ids=changed;weights=np.full(len(ids),1/len(v));value=values.mean()
                elif stat=='max':ids=np.array([values.argmax()]);weights=np.array([1.]);value=values[ids[0]]
                else:
                    order=np.argsort(values);pos=.95*(len(v)-1);lo=int(np.floor(pos));hi=int(np.ceil(pos));f=pos-lo;ids=np.array([order[lo],order[hi]]);weights=np.array([1-f,f]);value=values[ids]@weights
                scale=max(c['g1'],.0005 if c['metric']=='normalized_hand_error' else .05);row=np.zeros(D)
                for i,weight in zip(ids,weights):
                    item=instant.get((c['fps'],int(orig[i])))
                    if item is None:continue
                    grad=item[1][key]/scale*weight
                    blend=(1-a[i])*B[l[i]]+a[i]*B[r[i]]
                    row.reshape(controls,self.dim)[:]+=blend[:,None]*grad[None,:]
                result.append((value-c['g1']-5e-7)/scale);matrix.append(row)
            result=np.array(result);matrix=np.array(matrix);cache.update(z=z.copy(),values=result,jac=matrix)
            return result,matrix
        # Native velocity inequalities, including the two fixed boundary poses.
        edges=np.array(sorted({e for f in frames for e in (int(f)-1,int(f)) if 0<=e<self.n-1}))
        joint=sparse.kron(sparse.csr_matrix(B[frames]),sparse.eye(self.dim)).toarray()
        velocity=sparse.kron(sparse.csr_matrix(B[edges+1]-B[edges]),sparse.eye(self.dim)).toarray()/np.tile(self.steps,len(edges))[:,None]
        A=np.pad(np.vstack([joint,velocity]),((0,0),(0,1)))
        baseline=(self.x0[edges+1]-self.x0[edges])/self.steps
        lower=np.r_[(self.lo-self.x0[frames]).ravel(),(-1-baseline).ravel()];upper=np.r_[(self.hi-self.x0[frames]).ravel(),(1-baseline).ravel()]
        thresholds=np.r_[np.full(len(joint),.09),np.full(len(velocity),.3)]
        selected=set(np.flatnonzero(np.minimum(-lower,upper)<thresholds).tolist())
        z0=np.r_[np.zeros(D),max(score['max_relative_excess'],0)+1e-6]
        def objective(z):return z[-1]+1e-3*np.sum(z[:-1]**2)
        def derivative(z):return np.r_[2e-3*z[:-1],1.]
        def con(z):return z[-1]-evaluate(z)[0]
        def con_jac(z):return np.c_[-evaluate(z)[1],np.ones(len(records))]
        def callback(z):
            v,_=evaluate(z);history.append(dict(iteration=len(history)+1,slack=float(z[-1]),exact_max_relative_excess=float(v.max()),seconds=time.time()-start))
            if len(history)%10==0:print('WINDOW',self.key,json.dumps(history[-1]),flush=True)
        rng=np.random.default_rng(61);check=z0.copy();check[:-1]=rng.normal(size=D)*.002;direction=rng.normal(size=D+1);direction/=np.linalg.norm(direction)
        values,jacobian=evaluate(check);epsilon=1e-6
        numerical=(evaluate(check+epsilon*direction)[0]-evaluate(check-epsilon*direction)[0])/(2*epsilon)
        analytic=jacobian@direction[:-1];gradient_error=float(abs(numerical-analytic).max())
        assert gradient_error<2e-3,('Window gradient mismatch',gradient_error)
        print('WINDOW_GRADIENT',gradient_error,flush=True)
        for cut_round in range(6):
            ids=np.array(sorted(selected));print('WINDOW_CUTS',cut_round,'linear',len(ids),flush=True)
            result=minimize(objective,z0,jac=derivative,method='SLSQP',bounds=Bounds(np.r_[np.full(D,-.12),0],np.r_[np.full(D,.12),np.inf]),constraints=[LinearConstraint(A[ids],lower[ids],upper[ids]),dict(type='ineq',fun=con,jac=con_jac)],callback=callback,options=dict(maxiter=180,ftol=1e-10))
            actual=A@result.x;violated=np.flatnonzero((actual<lower-1e-8)|(actual>upper+1e-8))
            new=set(violated.tolist())-selected
            if not new:break
            selected.update(new);z0=result.x.copy();print('WINDOW_ADD_PHYSICAL_CUTS',len(new),flush=True)
        full=self.x0+B@result.x[:-1].reshape(controls,self.dim)
        final,_,_=self.evaluate(full)
        info=dict(success=bool(result.success),message=str(result.message),frames=frames.tolist(),controls=controls,variables=D,iterations=int(result.nit),slack=float(result.x[-1]),score=final,history=history)
        print('WINDOW_FINAL',self.key,json.dumps({k:v for k,v in info.items() if k not in ('history','frames')}),flush=True)
        return full,info

    def verify(self):
        score,metrics,obs=self.evaluate(self.x0,True);discrepancy=[]
        for o in obs:
            c=o['check'];matches=[x for x in self.seed_report['checks'] if all(x.get(k,'')==c.get(k,'') for k in ('fps','section','side','metric','stat'))]
            assert len(matches)==1;discrepancy.append(abs(matches[0]['h2']-o['value']))
        assert max(discrepancy)<2e-4,max(discrepancy)
        # Norm/absolute-angle errors are nondifferentiable at an exact fit.
        # Check a deterministic nearby interior pose without changing the seed.
        ref=self.refs[120];frame=self.n//2;q=self.q0[frame].copy()
        rng=np.random.default_rng(47)
        q[7+self.js]=np.clip(q[7+self.js]+rng.normal(size=self.dim)*.002,self.lo+1e-5,self.hi-1e-5)
        v,g=self.instant(q,ref,frame,True);fdmax=0.
        for j in range(self.dim):
            qp=q.copy();qm=q.copy();qp[7+self.js[j]]+=1e-6;qm[7+self.js[j]]-=1e-6;vp,_=self.instant(qp,ref,frame,False);vm,_=self.instant(qm,ref,frame,False)
            for key in v:fdmax=max(fdmax,abs((vp[key]-vm[key])/2e-6-g[key][j]))
        assert fdmax<5e-4,fdmax
        return dict(seed_metric_max_error=max(discrepancy),jacobian_fd_max_error=fdmax,finite_difference_columns=self.dim),score

def body_poses(model,q,bodies):
    data=mujoco.MjData(model);positions=[];rotations=[]
    for value in q:
        data.qpos[:]=value;mujoco.mj_kinematics(model,data)
        positions.append(data.xpos[bodies].copy());rotations.append(data.xmat[bodies].reshape(len(bodies),3,3).copy())
    return np.array(positions),np.array(rotations)

def reference_bodies(p):
    return [b for side in ('Left','Right') for b in p.arm_bodies[side]]+[p.torso]

def refine_wrist_yaw(p,q,budget_deg=15.):
    """Closest bounded wrist yaw curve without changing arm point trajectories.

    Fixed roll/pitch/arm poses make geodesic hand bounds intervals in yaw.
    All native, 50Hz and 600Hz intervals are imposed before the final FK audit.
    """
    sides=[p.side] if p.side else ['Left','Right'];S=len(sides);joints=[p.names.index(s.lower()+'_wrist_yaw_joint') for s in sides];bodies=[p.arm_bodies[s][-1] for s in sides];N=p.n*S
    canonical=q[:,7+np.array(joints)].copy();jr=[];jc=[];jv=[];targets=[];ar=[];ac=[];av=[];lower=[];upper=[];before={};constant_axes=[]
    for j in joints:constant_axes.append(p.m.jnt_axis[1+j].copy())
    def row(entries,lo,hi):
        k=len(lower)
        for c,v in entries:ar.append(k);ac.append(c);av.append(v)
        lower.append(lo);upper.append(hi)
    for rate in (120,50,600):
        times=np.arange(int(np.floor(p.old[-1]*rate+1e-8))+1)/rate;sampled=sample(q,times);_,rot=body_poses(p.m,sampled,bodies);errors=[]
        for side,(name,j,axis) in enumerate(zip(sides,joints,constant_axes)):
            target=Slerp(p.old,R.from_matrix(p.refs[120]['arms'][name]['wrist']))(times).as_matrix();y=sampled[:,7+j];r0=rot[:,side]@R.from_rotvec(-y[:,None]*axis).as_matrix();relative=target.transpose(0,2,1)@r0
            k=np.einsum('i,nij,j->n',axis,relative,axis);c=np.trace(relative,axis1=1,axis2=2)-k;s=np.einsum('nij,ji->n',relative,skew(axis));radius=np.hypot(c,s);phi=np.arctan2(s,c);phi+=2*np.pi*np.round((y-phi)/(2*np.pi))
            error=(R.from_matrix(target).inv()*R.from_matrix(rot[:,side])).magnitude();errors.append(error)
            cap=np.minimum(np.deg2rad(budget_deg),error+np.deg2rad(1e-5));ratio=(1+2*np.cos(cap)-k)/np.maximum(radius,1e-12)
            if np.any(ratio>1+1e-8):return q,dict(success=False,reason='Fixed arm pose requires more hand correction than the budget',rate=rate,side=name,violations=int((ratio>1+1e-8).sum()),minimum_error_deg=float(np.rad2deg(np.arccos(np.clip((k+radius-1)/2,-1,1))).max()))
            width=np.arccos(np.clip(ratio,-1,1));left=np.minimum(np.floor(times*120+1e-9).astype(int),p.n-1);right=np.minimum(left+1,p.n-1);alpha=np.maximum(times*120-left,0.)
            for i in range(len(times)):
                entries=[(int(left[i]*S+side),float(1-alpha[i]))]
                if alpha[i]>1e-10:entries.append((int(right[i]*S+side),float(alpha[i])))
                row(entries,float(phi[i]-width[i]-1e-9),float(phi[i]+width[i]+1e-9));index=len(targets);weight=np.sqrt(120/rate)*np.sqrt(max(radius[i],.01)/2)
                for col,value in entries:jr.append(index);jc.append(col);jv.append(value*weight)
                targets.append(phi[i]*weight)
        before[str(rate)]=dict(max_error_deg=float(np.rad2deg(np.array(errors)).max()))
    for i in range(p.n):
        for side,j in enumerate(joints):
            margin=np.deg2rad(2)+1e-6;lo=p.m.jnt_range[1+j,0]+margin;hi=p.m.jnt_range[1+j,1]-margin;row([(i*S+side,1.)],float(lo),float(hi))
    for i in range(p.n-1):
        for side,j in enumerate(joints):
            step=p.speeds[j]/120*.99999;row([(i*S+side,-1.),((i+1)*S+side,1.)],float(-step),float(step))
    J=sparse.coo_matrix((jv,(jr,jc)),shape=(len(targets),N)).tocsc();P=J.T@J+sparse.eye(N,format='csc')*1e-8;g=-np.asarray(J.T@np.array(targets)).ravel()-1e-8*canonical.ravel();A=sparse.coo_matrix((av,(ar,ac)),shape=(len(lower),N)).tocsc();solver=osqp.OSQP()
    solver.setup(P=sparse.triu(P).tocsc(),q=g,A=A,l=np.array(lower),u=np.array(upper),eps_abs=1e-8,eps_rel=1e-8,max_iter=20000,polishing=True,verbose=False);result=solver.solve()
    if result.x is None or not np.isfinite(result.x).all() or result.info.status not in ('solved','solved inaccurate','maximum iterations reached'):return q,dict(success=False,reason=result.info.status,before=before)
    final=q.copy();final[:,7+np.array(joints)]=result.x.reshape(p.n,S);after={};max_point_drift=0.
    for rate in (120,50,600):
        times=np.arange(int(np.floor(p.old[-1]*rate+1e-8))+1)/rate;pp,rr=body_poses(p.m,sample(final,times),bodies);oldp,oldr=body_poses(p.m,sample(q,times),bodies);errors=[];old_errors=[]
        for side,name in enumerate(sides):
            target=Slerp(p.old,R.from_matrix(p.refs[120]['arms'][name]['wrist']))(times)
            errors.append((target.inv()*R.from_matrix(rr[:,side])).magnitude());old_errors.append((target.inv()*R.from_matrix(oldr[:,side])).magnitude())
        error=np.array(errors);old_error=np.array(old_errors);max_point_drift=max(max_point_drift,float(np.linalg.norm(pp-oldp,axis=-1).max()))
        after[str(rate)]=dict(max_error_deg=float(np.rad2deg(error).max()),mean_error_deg=float(np.rad2deg(error).mean()),pointwise_increase_deg=float(np.rad2deg(error-old_error).max()))
    yaw=final[:,7+np.array(joints)];margin=np.deg2rad(2);lo=p.m.jnt_range[1+np.array(joints),0]+margin;hi=p.m.jnt_range[1+np.array(joints),1]-margin
    speed=float((abs(np.diff(yaw,axis=0))*120/p.speeds[joints]).max());range_excess=max(float((lo-yaw).max()),float((yaw-hi).max()),0.)
    success=max_point_drift<1e-9 and speed<=1+1e-6 and range_excess<=1e-8 and all(z['max_error_deg']<=budget_deg+1e-5 and z['pointwise_increase_deg']<=3e-5 for z in after.values())
    return (final if success else q),dict(success=success,qp_status=result.info.status,optimization_converged=result.info.status in ('solved','solved inaccurate'),yaw_speed_ratio=speed,yaw_range_excess_rad=range_excess,before=before,after=after,arm_point_max_drift_m=max_point_drift,budget_deg=budget_deg,scope='Yaw-only optimization; primary arm/torso geometry is invariant, and sampled hand orientation increases are limited to numerical tolerance.')

def load_reference(p,path):
    with np.load(path,allow_pickle=False) as z:meta=json.loads(str(z['metadata']));q=z['qpos'].copy()
    assert meta['key']==p.key and meta['model_sha256']==digest(BUNDLE/'code/gear_sonic/data/assets/robot_description/mjcf/h2.xml')
    assert meta['raw_trace_sha256']==digest(Path(p.job['work'])/'h2'/f'{p.key}_trace.npz') and q.shape==p.q0.shape
    protected=np.r_[np.arange(7),[7+i for i,n in enumerate(p.names) if any(s in n for s in ('hip','knee','ankle'))]]
    assert np.array_equal(q[:,protected],p.q0[:,protected]),'Reference changed root or legs'
    return meta,q

def attach_reference(p,path):
    meta,q=load_reference(p,path);bodies=reference_bodies(p)
    for ref in p.refs.values():ref['fixed_position'],ref['fixed_rotation']=body_poses(p.m,sample(q,ref['tt']),bodies)
    p.fixed_goal=dict(path=Path(path),bodies=bodies,metadata=meta)

def raw_quality_review(candidate,baseline):
    checks=[];wrist=[]
    for rate in ('120','50'):
        for side in ('Left','Right'):
            for metric in ('upper_direction_error_deg','forearm_direction_error_deg','elbow_flex_error_deg','normalized_hand_error','wrist_orientation_error_deg'):
                for stat in ('mean','p95','max'):
                    old=baseline['audit_h2'][rate]['arms'][side][metric][stat];new=candidate['audit_h2'][rate]['arms'][side][metric][stat]
                    row=dict(rate=int(rate),section='arms',side=side,metric=metric,stat=stat,baseline=old,candidate=new,passed=new<=old+1e-6)
                    (wrist if metric=='wrist_orientation_error_deg' else checks).append(row)
        for metric in ('torso_orientation_error_deg','torso_lateral_error_deg'):
            for stat in ('mean','p95','max'):
                old=baseline['audit_h2'][rate][metric][stat];new=candidate['audit_h2'][rate][metric][stat]
                checks.append(dict(rate=int(rate),section='torso',metric=metric,stat=stat,baseline=old,candidate=new,passed=new<=old+1e-6))
    angular=[r for r in checks if r['stat']=='mean' and r['metric']!='normalized_hand_error']
    before=float(sum(r['baseline'] for r in angular));after=float(sum(r['candidate'] for r in angular))
    return dict(primary_checks=checks,primary_regressions=[r for r in checks if not r['passed']],primary_nonregression_pass=all(r['passed'] for r in checks),primary_mean_angular_before=before,primary_mean_angular_after=after,primary_improvement_pass=after<before-1e-4,raw_wrist_checks=wrist,raw_wrist_regressions=[r for r in wrist if not r['passed']])

def fixed_reference_check(p,candidate,path):
    meta,reference=load_reference(p,path);_,_,_,q=load(candidate,p);bodies=reference_bodies(p);rates={}
    for rate in (120,50,600):
        times=np.arange(int(np.floor(p.old[-1]*rate+1e-8))+1)/rate
        cp,cr=body_poses(p.m,sample(q,times),bodies);gp,gr=body_poses(p.m,sample(reference,times),bodies)
        position=np.linalg.norm(cp-gp,axis=-1)*1000
        ids=[2,5,6];rot=(R.from_matrix(gr[:,ids].reshape(-1,3,3)).inv()*R.from_matrix(cr[:,ids].reshape(-1,3,3))).magnitude()*180/np.pi
        raw_wrist=[];goal_wrist=[]
        for i,side in zip((2,5),('Left','Right')):
            target=Slerp(p.old,R.from_matrix(p.refs[120]['arms'][side]['wrist']))(times)
            raw_wrist.append((target.inv()*R.from_matrix(cr[:,i])).magnitude()*180/np.pi);goal_wrist.append((target.inv()*R.from_matrix(gr[:,i])).magnitude()*180/np.pi)
        raw_max=float(np.max(raw_wrist));goal_max=float(np.max(goal_wrist));budget=meta.get('wrist_correction_budget_deg');budget_pass=budget is None or (raw_max<=budget+1e-5 and goal_max<=budget+1e-5)
        rates[str(rate)]=dict(samples=len(times),max_body_position_error_mm=float(position.max()),max_wrist_torso_rotation_error_deg=float(rot.max()),raw_wrist_max_deg=raw_max,goal_raw_wrist_max_deg=goal_max,wrist_correction_budget_deg=budget,wrist_budget_pass=budget_pass,passed=bool(position.max()<=.5+1e-6 and rot.max()<=.1+1e-6 and budget_pass))
    return dict(reference=str(path),reference_sha256=digest(path),goal_metadata=meta,rates=rates,passed=all(r['passed'] for r in rates.values()),scope='Fidelity to a frozen reconstructed H2 target; raw BVH quality is checked separately.')

def peak_diagnostics(p,candidate,independent,count):
    _,_,_,q=load(candidate,p);_,base,_=p.evaluate(p.x0,False);_,current,_=p.evaluate(q[:,7+p.js],False);out=[];agreement=[]
    for rate in (120,50):
        keys=[('arms',side,metric) for side in ('Left','Right') for metric in ('upper_direction_error_deg','forearm_direction_error_deg','elbow_flex_error_deg','wrist_orientation_error_deg')]+[('torso','',metric) for metric in ('torso_orientation_error_deg','torso_lateral_error_deg')]
        for section,side,metric in keys:
            v,_,l,r,a=current[(rate,section,side,metric)];old=base[(rate,section,side,metric)][0]
            exact=independent['audit_h2'][str(rate)]['arms'][side][metric]['max'] if section=='arms' else independent['audit_h2'][str(rate)][metric]['max']
            agreement.append(abs(float(v.max())-exact))
            ids=np.argsort(v)[-count:][::-1]
            for i in ids:
                frames=np.unique([l[i],r[i]]);angles=q[frames][:,7+p.js];margin=np.minimum(angles-p.lo,p.hi-angles).min(0);nearest=np.argsort(margin)[:3]
                edges=np.unique(np.clip(np.r_[frames-1,frames],0,p.n-2));speed=abs(q[edges+1,7:]-q[edges,7:])*120/p.speeds;ei,ji=np.unravel_index(speed.argmax(),speed.shape)
                out.append(dict(rate=rate,section=section,side=side,metric=metric,sample=int(i),time_seconds=float(p.refs[rate]['tt'][i]),native_frames=frames.tolist(),interpolation_alpha=float(a[i]),candidate=float(v[i]),baseline_same_sample=float(old[i]),baseline_max=float(old.max()),closest_effective_joint_bounds=[dict(name=p.names[p.js[j]],margin_deg=float(np.rad2deg(margin[j]))) for j in nearest],nearby_max_speed_ratio=float(speed[ei,ji]),nearby_max_speed_joint=p.names[ji],nearby_max_speed_native_edge=int(edges[ei])))
    assert max(agreement)<2e-4,('Peak locator disagrees with independent auditor',max(agreement))
    return dict(independent_max_metric_agreement=max(agreement),peaks=out)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--keys',nargs='+',default=KEYS);ap.add_argument('--iterations',type=int,default=30);ap.add_argument('--window-key');ap.add_argument('--knots',type=int);ap.add_argument('--side',choices=['Left','Right']);ap.add_argument('--initial',type=Path);ap.add_argument('--progress',type=Path,default=ROOT/'unresolved_progress.json');ap.add_argument('--reconstruct',action='store_true');ap.add_argument('--export-reference',type=Path);ap.add_argument('--reference',type=Path);ap.add_argument('--fit-reference',type=Path);ap.add_argument('--audit-csv',type=Path);ap.add_argument('--worst-frames',type=int,default=3);ap.add_argument('--continue-branch-json',type=Path);ap.add_argument('--quality-window',action='store_true');ap.add_argument('--temporary-only',action='store_true');ap.add_argument('--anatomical',action='store_true');ap.add_argument('--wrist-budget-deg',type=float,default=15.);ap.add_argument('--arms-only',action='store_true');a=ap.parse_args()
    common=module('upper_sqp_common',SCRIPTS/'run_h2_foot_preserving_reopt.py');api=common.api_for(BUNDLE);signature=api.verify_bundle();jobs={j['key']:j for j in api.jobs_for(SOURCE)}
    auditor=module('upper_sqp_auditor',SCRIPTS/'h2_contact_checked_foot_audit.py')
    if a.export_reference:
        assert a.reference and not a.reference.exists(),'Use a new --reference path'
        key=a.export_reference.stem;p=Problem(key,jobs[key],initial=a.export_reference)
        meta=dict(contract='frozen_projected_h2_upper_v1',key=key,source_csv=str(a.export_reference),source_csv_sha256=digest(a.export_reference),raw_trace_sha256=digest(Path(jobs[key]['work'])/'h2'/f'{key}_trace.npz'),model_sha256=digest(BUNDLE/'code/gear_sonic/data/assets/robot_description/mjcf/h2.xml'),fps=120,quality_review_stage='pending_raw_source_error_review',note='Projected target frozen before refitting. Fidelity does not establish raw BVH motion quality.')
        meta['wrist_correction_budget_deg']=a.wrist_budget_deg
        meta['generation_details']=p.seed_report.get('anatomical_reconstruction')
        times=np.arange((p.n-1)*5+1)/600.;_,rot=body_poses(p.m,sample(p.q0,times),[p.arm_bodies[s][-1] for s in ('Left','Right')]);maximum=0.
        for i,side in enumerate(('Left','Right')):
            target=Slerp(p.old,R.from_matrix(p.refs[120]['arms'][side]['wrist']))(times);maximum=max(maximum,float((target.inv()*R.from_matrix(rot[:,i])).magnitude().max()*180/np.pi))
        assert maximum<=a.wrist_budget_deg+1e-5,('Reference exceeds raw wrist correction budget',maximum,a.wrist_budget_deg)
        meta['dense_raw_wrist_max_deg']=maximum
        a.reference.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.reference,qpos=p.q0,metadata=np.array(json.dumps(meta)));print('FROZEN_REFERENCE',a.reference,flush=True);return
    if a.audit_csv:
        key=a.audit_csv.stem;p=Problem(key,jobs[key]);scratch=Path(tempfile.mkdtemp(prefix='h2_independent_audit_',dir='/tmp'));path=scratch/'refined/csv'/f'{key}.csv';path.parent.mkdir(parents=True);shutil.copy2(a.audit_csv,path)
        auditor.audit_task((str(BUNDLE),str(SOURCE),str(scratch),dict(jobs[key],recipe='independent_cli_audit')))
        report=json.loads((scratch/'reports'/f'{key}.json').read_text());review=raw_quality_review(report,p.seed_report)
        _,_,_,q=load(path,p);protected=[i for i in range(p.m.nq) if i not in set(7+p.js)];protected_exact=np.array_equal(q[:,protected],p.q0[:,protected])
        fidelity=fixed_reference_check(p,path,a.reference) if a.reference else None
        final=bool(report['status']=='accepted' or (fidelity and fidelity['passed'] and report['physical_and_margin_pass'] and protected_exact and review['primary_nonregression_pass'] and review['primary_improvement_pass']))
        result=dict(key=key,candidate=str(a.audit_csv),candidate_sha256=digest(a.audit_csv),legacy_full_pass=report['status']=='accepted',legacy_failed_checks=[c for c in report['checks'] if not c['passed']],physical_pass=report['physical_and_margin_pass'],root_legs_exact=bool(protected_exact),foot_guard=report['trajectory_guard'],fixed_reference=fidelity,raw_quality_review=review,final_pass=final,report=str(scratch/'reports'/f'{key}.json'),training_admission_performed=False)
        result['validation_scope']='Kinematic FK, joint ranges, 2-degree shoulder/wrist margins and native URDF speeds. Dynamics and comprehensive self-collision are outside this predicate.'
        if a.worst_frames>0:result['peak_diagnostics']=peak_diagnostics(p,path,report,a.worst_frames)
        output=scratch/'criteria.json';save(output,result);print(json.dumps(dict(key=key,legacy_full_pass=result['legacy_full_pass'],physical_pass=result['physical_pass'],root_legs_exact=result['root_legs_exact'],fixed_reference_pass=fidelity['passed'] if fidelity else None,raw_primary_nonregression_pass=review['primary_nonregression_pass'],primary_regressions=len(review['primary_regressions']),raw_wrist_regressions=len(review['raw_wrist_regressions']),final_pass=final,criteria_json=str(output)),ensure_ascii=False),flush=True)
        raise SystemExit(0 if final else 2)
    if a.window_key:
        p=Problem(a.window_key,jobs[a.window_key],a.side,a.initial)
        if a.quality_window:
            assert a.initial
            p.window_padding=12
            canonical=json.loads((ROOT/'existing_eight_reaudit/reports'/f'{p.key}.json').read_text());custom=[]
            for c in p.checks:
                if c['metric']=='wrist_orientation_error_deg':continue
                profile=canonical['audit_h2'][str(c['fps'])]
                if c['section']=='arms':profile=profile['arms'][c['side']]
                value=profile[c['metric']][c['stat']]
                custom.append(dict(c,g1=value*.9995,cap_source='canonical_H2_raw_BVH_quality_with_0p05_percent_headroom'))
            p.checks=custom
        test,_=p.verify();x,info=p.window(a.knots);scratch=Path(tempfile.mkdtemp(prefix='h2_upper_window_',dir='/tmp'));path=scratch/'refined/csv'/f'{p.key}.csv';path.parent.mkdir(parents=True)
        raw=p.raw.copy();raw[:,p.cols]=p.assemble(x)[:,7:]
        with path.open('w',newline='') as f:w=csv.writer(f);w.writerow(p.header);w.writerows(raw)
        auditor.audit_task((str(BUNDLE),str(SOURCE),str(scratch),dict(jobs[p.key],recipe='upper_nonlinear_window')))
        report=json.loads((scratch/'reports'/f'{p.key}.json').read_text())
        if report['status']=='accepted':
            out=ROOT/'resolved_upper';dest=out/'refined/csv'/f'{p.key}.csv';dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest);rp=out/'reports'/f'{p.key}.json';rp.parent.mkdir(parents=True,exist_ok=True);report['validated_source_output']=report['output'];report['output']=str(dest);save(rp,report)
        label=p.key+('_'+a.side if a.side else '')+('_quality' if a.quality_window else '')
        save(Path('/tmp')/f'h2_upper_window_{label}_result.json',dict(key=p.key,side=a.side,status=report['status'],tests=test,optimization=info,failed_checks=[c for c in report['checks'] if not c['passed']],best_csv=str(path),candidate_sha256=digest(path),accepted_csv=str(dest) if report['status']=='accepted' else None))
        print('WINDOW_AUDIT',p.key,report['status'],sum(not c['passed'] for c in report['checks']),flush=True)
        return
    pins=json.loads((ROOT/'existing_eight_reaudit/summary.json').read_text())['training_catalog_hashes_unchanged']
    progress_path=a.progress;prior=progress_path if progress_path.exists() else ROOT/'unresolved_progress.json';progress=json.loads(prior.read_text()) if prior.exists() else dict(motions={},training_started=False)
    scratch=Path(tempfile.mkdtemp(prefix='h2_upper_sqp_',dir='/tmp'));print('SCRATCH',scratch,flush=True)
    for key in a.keys:
        assert key in KEYS;assert api.raw_valid(Path(jobs[key]['work']),jobs[key])
        initial=a.initial if a.initial and a.initial.stem==key else None
        p=Problem(key,jobs[key],side=a.side if (a.continue_branch_json or a.anatomical) else None,initial=initial,arms_only=a.anatomical or a.arms_only);test,score=p.verify();branch=None
        if a.continue_branch_json:
            diagnosis=next(r for r in json.loads(a.continue_branch_json.read_text())['results'] if r['key']==key);branch=p.continue_branch(diagnosis)
        x=p.x0.copy();history=[];begin=time.time()
        if a.fit_reference:attach_reference(p,a.fit_reference)
        print('CHECK',key,json.dumps(test),score,flush=True)
        reconstruction=None
        if a.anatomical:x,reconstruction=p.anatomical_project(a.iterations,a.wrist_budget_deg,scratch/f'{key}_anatomical_checkpoint.npz');history=reconstruction['history']
        elif a.reconstruct or a.fit_reference or a.continue_branch_json:x,reconstruction=p.reconstruct(a.iterations);history=reconstruction['history']
        for it in range(0 if (a.reconstruct or a.fit_reference or a.continue_branch_json or a.anatomical) else a.iterations):
            score,metrics,observed=p.evaluate(x,True)
            if score['failed']==0 and score['physical']:break
            delta,info=p.qp(x,metrics);won=False;trials=[]
            if delta is not None:
                for alpha in (1.,.5,.25,.125,.0625,.03125):
                    trial=np.clip(x+alpha*delta,p.lo,p.hi);candidate,_,new_observed=p.evaluate(trial)
                    broken=sum(old['over']==0 and new['over']>0 for old,new in zip(observed,new_observed))
                    trials.append(dict(alpha=alpha,score=candidate,passed_metrics_broken=int(broken)))
                    if not broken and candidate['physical'] and candidate['cost']<score['cost']-1e-11:x=trial;score=candidate;won=True;break
            row=dict(iteration=it+1,score=score,qp=info,accepted=won,seconds=time.time()-begin,trials=trials);history.append(row);print(key,json.dumps({k:v for k,v in row.items() if k!='trials'}),flush=True)
            progress['motions'][key]=dict(stage='optimizing',initial_failed=len(p.seed_report['failed_checks']) if 'failed_checks' in p.seed_report else sum(not z['passed'] for z in p.seed_report['checks']),last=row,tests=test)
            save(progress_path,progress)
            if not won:break
        q=p.assemble(x);raw=p.raw.copy();raw[:,p.cols]=q[:,7:]
        protected=[i for i in range(p.m.nq) if i not in set(7+p.js)];assert np.array_equal(q[:,protected],p.q0[:,protected])
        path=scratch/'refined/csv'/f'{key}.csv';path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('w',newline='') as f:w=csv.writer(f);w.writerow(p.header);w.writerows(raw)
        auditor.audit_task((str(BUNDLE),str(SOURCE),str(scratch),dict(jobs[key],recipe='upper_elastic_sqp')))
        report=json.loads((scratch/'reports'/f'{key}.json').read_text());failed=[c for c in report['checks'] if not c['passed']]
        new_reference_pass=False;quality=None
        if reconstruction:
            # Original independent auditor reconstructs raw BVH metrics itself.
            # Never score against FK targets exported by this candidate.
            def raw_mean_score(r):
                values=[]
                for rate in ('120','50'):
                    d=r['audit_h2'][rate];v=d['torso_orientation_error_deg']['mean']/.25
                    for side in ('Left','Right'):
                        z=d['arms'][side];v+=z['upper_direction_error_deg']['mean']+z['forearm_direction_error_deg']['mean']+z['elbow_flex_error_deg']['mean']*.2+z['wrist_orientation_error_deg']['mean']/5.
                    values.append(v)
                return float(np.mean(values))
            canonical=json.loads((ROOT/'existing_eight_reaudit/reports'/f'{key}.json').read_text());before=raw_mean_score(canonical);after=raw_mean_score(report)
            foot_exact=all(z['max_position_drift_mm']<1e-6 and z['max_rotation_drift_deg']<1e-6 for z in report['trajectory_guard']['rates'].values())
            review=raw_quality_review(report,canonical);fidelity=fixed_reference_check(p,path,a.fit_reference) if a.fit_reference else None
            new_reference_pass=bool(fidelity and fidelity['passed'] and report['physical_and_margin_pass'] and foot_exact and review['primary_nonregression_pass'] and review['primary_improvement_pass'])
            quality=dict(before=before,after=after,improvement_percent=(before-after)/before*100,definition='Mean of 120/50Hz raw-BVH direction + 0.2*flex + wrist/5 + torso/0.25 angular errors; regenerated by unchanged independent auditor; no exported FK target used.')
            report['h2_reference_reconstruction']=dict(contract='bvh_h2_upper_joint_feasible_v1',validation_pass=new_reference_pass,raw_source_quality=quality,source_goal_weights=reconstruction['source_weights'],projection_cost_before=reconstruction['before'],projection_cost_after=reconstruction['after'],iterations=len(history),source_trace_sha256=digest(Path(jobs[key]['work'])/'h2'/f'{key}_trace.npz'),seed_sha256=digest(initial or p.seed),root_and_legs_exact=True,feet_exact=foot_exact,legacy_g1_pass=not failed,legacy_status=report['status'],quality_compared_to_candidate_exported_pose=False,training_started=False)
            report['h2_reference_reconstruction'].update(raw_quality_review=review,fixed_reference=fidelity,stage='refitted_candidate' if fidelity else 'projection_candidate')
            if branch:report['h2_reference_reconstruction']['branch_continuation']=branch
            save(scratch/'reports'/f'{key}.json',report)
        persist=(report['status']=='accepted' or new_reference_pass) and not a.temporary_only
        if persist:
            out=ROOT/'resolved_upper';dest=out/'refined/csv'/f'{key}.csv';dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(Path(report['output']),dest)
            if new_reference_pass and a.fit_reference:
                target=out/'references'/f'{key}.npz';target.parent.mkdir(parents=True,exist_ok=True)
                if a.fit_reference.resolve()!=target.resolve():shutil.copy2(a.fit_reference,target)
                assert digest(target)==digest(a.fit_reference)
                report['h2_reference_reconstruction']['fixed_reference']['reference']=str(target)
            rp=out/'reports'/f'{key}.json';rp.parent.mkdir(parents=True,exist_ok=True);report['validated_source_output']=report['output'];report['output']=str(dest);save(rp,report)
        progress['motions'][key]=dict(stage='audited',status='reconstructed_reference_validated' if new_reference_pass else report['status'],legacy_status=report['status'],failed_checks=failed,physical=report['physical_and_margin_pass'],foot_guard=report['trajectory_guard']['passed'],tests=test,history=history,scratch=str(scratch),best_csv=str(path),candidate_sha256=digest(path),accepted_csv=str(dest) if persist else None,new_reference_pass=new_reference_pass,raw_source_quality=quality)
        save(progress_path,progress);print('FINAL',key,report['status'],'failed',len(failed),'new_reference',new_reference_pass,'quality',quality,flush=True)
    assert all(digest(p)==h for p,h in pins.items())
    progress['training_catalog_hashes_unchanged']=pins;save(progress_path,progress)

if __name__=='__main__':main()
