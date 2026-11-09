"""Verified-neighbor residual proposals inside the original optimization domain."""
from pathlib import Path
import time
import numpy as np
from scipy import sparse
from scipy.ndimage import gaussian_filter1d
if __package__:
    from .solution_bank import Bank,foot_features,arm_features,yaw,local_xy
else:
    from solution_bank import Bank,foot_features,arm_features,yaw,local_xy
if __package__:
    from .adaptive_stopping import finite
else:
    from adaptive_stopping import finite

POLICY=dict(format='retrieved_verified_residual_initialization_v1',
    source_targets_changed=False,admission_changed=False,original_variable_bounds_preserved=True,
    same_key_and_mirror_excluded=True,lower_neighbors=[8,1],upper_neighbors=8,
    no_motion_wall_limit=True,repair_candidates=3,repair_max_statistical_regressions=36,
    repair_min_relative_cost_gain=.01,fast_goal_minimum_gain_removed=True,
    torso_canonical_nonregression=True,native_residual_steps=6,foot_contact_step_m=.01,
    elastic_contact_proposals=True,hard_attempt=True,source_arm_section_qualified=True,
    restore_final_statistics_only=True,residual_sqp_steps=6,contact_slack_weight=1e7)

def project_steps(values,lo,hi,step):
    result=np.clip(values,lo,hi).copy()
    for i in range(1,len(result)):result[i]=np.clip(result[i],np.maximum(lo,result[i-1]-step),np.minimum(hi,result[i-1]+step))
    for i in range(len(result)-2,-1,-1):result[i]=np.clip(result[i],np.maximum(lo,result[i+1]-step),np.minimum(hi,result[i+1]+step))
    return result

class NoReusableGoal(ValueError):pass

class Reuse:
    def __init__(self,batch,out,key,save,bank_path):
        self.batch,self.out,self.key,self.save=batch,Path(out),key,save
        self.bank=Bank(bank_path,verify_hash=False);self.events=[];self.disabled=False;self.fast=True;self.goal_ready=False
        self.repair_fixed=None
        model=batch.BUNDLE/'code/gear_sonic/data/assets/robot_description/mjcf/h2.xml'
        assert self.bank.meta['model_sha256']==batch.sha(model)

    def record(self,event):
        self.events.append(finite(event));self.save(self.out/'jobs'/self.key/'retrieval.json',self.events)

    def donors(self,owner):
        ids,count=np.unique(owner[owner>=0],return_counts=True)
        chosen=sorted(zip(ids,count),key=lambda x:-x[1])[:5]
        return [dict(key=self.bank.meta['donor_keys'][int(i)],matched_samples=int(n)) for i,n in chosen]

    def vertical_project(self,p,x):
        """With joints fixed, sole heights are affine in the root-Z curve."""
        q=p.assemble(x);times=np.arange((p.n-1)*5+1)/600.
        _,_,heights=p.fm.poses(self.batch.F.sample(q,times));_,_,lift=p.goals(times)
        lower=(p.floor+p.margin-heights).max(1)
        upper=np.where(lift>1e-8,p.floor+.0004-heights,np.inf).min(1)
        strict_contact=not np.any(lower>upper+1e-9)
        left=np.minimum(np.floor(times*120+1e-9).astype(int),p.n-1);right=np.minimum(left+1,p.n-1);alpha=np.maximum(0.,times*120-left)
        rows=np.repeat(np.arange(len(times)),2);cols=np.c_[left,right].ravel();data=np.c_[1-alpha,alpha].ravel()
        C=sparse.coo_matrix((data,(rows,cols)),shape=(len(times),p.n)).tocsc();C.eliminate_zeros()
        I=sparse.eye(p.n,format='csc');D=sparse.diags([np.ones(p.n-2),-2*np.ones(p.n-2),np.ones(p.n-2)],[0,1,2],shape=(p.n-2,p.n),format='csc')
        A=sparse.vstack([C,I],format='csc');lo=np.r_[lower,p.low[:,2]-x[:,2]];hi=np.r_[upper if strict_contact else np.full(len(upper),np.inf),p.high[:,2]-x[:,2]]
        grounded=np.isfinite(upper);weight=np.where(grounded,1.,.01)
        target=np.where(grounded,np.minimum(upper,lower+.0001),0.)
        weighted=C.multiply(weight[:,None]);P=C.T@weighted+.01*I+2*(D.T@D);linear=-np.asarray(C.T@(weight*target)).ravel()
        import osqp
        solver=osqp.OSQP();solver.setup(P=sparse.triu(P).tocsc(),q=linear,A=A,l=lo,u=hi,
            eps_abs=1e-8,eps_rel=1e-8,max_iter=5000,polishing=True,verbose=False)
        try:result=solver.solve_interior() if hasattr(solver,'solve_interior') else solver.solve()
        except Exception:return None
        if result.x is None or not np.isfinite(result.x).all():return None
        trial=x.copy();trial[:,2]=np.clip(trial[:,2]+result.x,p.low[:,2],p.high[:,2])
        if np.max(lo-A@result.x)>1e-6 or np.max(A@result.x-hi)>1e-6:return None
        return trial

    def lower(self,p,initial):
        if self.disabled:return initial
        began=time.monotonic();baseline=p.measure(initial)
        if not baseline['feasible']:return initial
        for folder in ('fixed_panel_run_v2_20261106','fixed_panel_run_v1_20261106','hard_retry_check_v1','reuse_full_batch_v4','group_retry_holdout_v1','group_retry_check_v1','reuse_full_batch_v3'):
            directory=Path('/home/arrc/lib/groot_wbc/artifacts/h2_held_fast_setup_20261104')/folder/'official/diagnostics'/self.key
            path=directory/'latest.csv';report=directory/'latest_audit.json'
            if not path.exists() or not report.exists():continue
            receipt=self.batch.read(report)
            if receipt.get('refined_sha256')!=self.batch.sha(path):continue
            _,_,_,q=self.batch.F.load(path,p.fm)
            if q.shape!=p.q0.shape or not np.array_equal(q[:,3:7],p.q0[:,3:7]):continue
            cached=np.c_[q[:,:3],q[:,7+p.js]]
            score=p.measure(cached)
            if score['feasible'] and score['grounded_target_height_max_mm']<baseline['grounded_target_height_max_mm']-.01:
                self.record(dict(stage='lower',method='resume_verified_previous_lower_candidate',applied=True,
                    path=str(path),candidate_sha256=self.batch.sha(path),before=finite(baseline),after=finite(score),
                    original_bounds_preserved=True,final_admission_required=True))
                return cached
        height=baseline['grounded_target_height_max_mm']
        if not np.isfinite(height) or height<=.45:return initial
        protected=initial.copy();q=p.original
        actual_position,actual_rotation,_=p.fm.poses(q)
        gp,gr,lift=p.goals(p.old)
        feature=foot_features(q,p.js,actual_position,actual_rotation,gp,gr)
        contact=(lift[:,0]>1e-8).astype(int)+2*(lift[:,1]>1e-8).astype(int)
        winner=initial;score=baseline;trials=[];used=[];winner_donors=[];winning_method='original_seed';winning_delta=0.
        for neighbors in [101,108]+POLICY['lower_neighbors']+[0]:
            if neighbors:
                delta,distance,owner=self.bank.query('lower',feature,self.key,contact=contact,k=neighbors%100,absolute=neighbors>=100)
            else:
                delta=np.zeros((p.n,15));distance=np.zeros(p.n);owner=np.full(p.n,-1)
            confidence=np.isfinite(distance)&(distance<=6.)
            delta=gaussian_filter1d(delta,sigma=2.,axis=0,mode='nearest');delta[~confidence]=0.
            used=self.donors(owner)
            if not np.any(confidence):continue
            if neighbors<100:delta[:,:3]=local_xy(delta[:,:3],yaw(q))
            for amount in (1.,.5):
                trial=initial.copy()
                if neighbors>=100:
                    trial[:,2]=np.clip((1-amount)*initial[:,2]+amount*delta[:,2],p.low[:,2],p.high[:,2])
                else:trial[:,:3]=np.clip(p.original[:,:3]+amount*delta[:,:3],p.low[:,:3],p.high[:,:3])
                if neighbors:
                    import fast_primitives
                    desired=np.clip((1-amount)*initial[:,3:]+amount*delta[:,3:] if neighbors>=100 else p.original[:,7+p.js]+amount*delta[:,3:],p.qlo,p.qhi)
                    trial[:,3:]=fast_primitives.native_curve(desired,p.qlo,p.qhi,p.step*.9995)
                else:
                    trial[:,3:]=initial[:,3:]
                trial=self.vertical_project(p,trial)
                if trial is None:
                    trials.append(dict(neighbors=neighbors,amount=amount,rejected='vertical_domain_infeasible'));continue
                candidate=p.measure(trial)
                trials.append(dict(neighbors=neighbors,amount=amount,score=finite(candidate)))
                candidate_height=candidate['grounded_target_height_max_mm'];old_height=score['grounded_target_height_max_mm']
                improves=(candidate_height<old_height-.01 or (candidate_height<=.45 and candidate['cost']<score['cost']-1e-5))
                if candidate['feasible'] and improves:
                    winner=trial;score=candidate;winner_donors=used;winning_method='retrieved_validated_pose' if neighbors>=100 else 'retrieved_pose_residual' if neighbors else 'analytic_vertical_initialization';winning_delta=float(np.max(abs(delta[:,3:])))
                if score['grounded_target_height_max_mm']<=.45:break
            if score['grounded_target_height_max_mm']<=.45:break
        # The original anchor/bounds/upper/root rotation/targets stay untouched.
        assert np.array_equal(initial,protected)
        assembled=p.assemble(winner);before=p.assemble(initial)
        protected_cols=[i for i in range(assembled.shape[1]) if i>=3 and i not in (7+p.js)]
        assert np.array_equal(assembled[:,protected_cols],before[:,protected_cols])
        applied=not np.array_equal(winner,initial)
        self.record(dict(stage='lower',applied=applied,seconds=time.monotonic()-began,
            before=finite(baseline),after=finite(score),donors=winner_donors,trials=trials,method=winning_method,retrieved_joint_delta_max_rad=winning_delta,
            same_key_and_mirror_excluded=True,original_bounds_preserved=True,final_admission_required=True))
        return winner

    def preserve_refit(self,p,trial):
        if not hasattr(p,'_retrieval_exact_columns'):
            with np.load(p.fixed_goal['path']) as z:goal=z['qpos'][:,7+p.js]
            p._retrieval_exact_columns=np.all(goal==p.x0,axis=0)
        result=trial.copy();result[:,p._retrieval_exact_columns]=p.x0[:,p._retrieval_exact_columns]
        return result

    @staticmethod
    def original_stats(p,stats):
        """Compare generation with the same immutable source as the final audit."""
        result={k:dict(v) for k,v in stats.items()}
        for key in result:
            rate,section,side,name=key
            audit=p.original.get('audit_h2',{}).get(str(rate),{})
            original=audit.get('arms',{}).get(side,{}).get(name) if section=='arms' else audit.get(name)
            if original is not None:
                result[key]={s:float(original[s]) for s in ('mean','p95','max')}
        return result

    @staticmethod
    def torso_nonregression(p,metrics):
        for rate in (120,50):
            original=p.original['audit_h2'][str(rate)]
            for name in ('torso_orientation_error_deg','torso_lateral_error_deg'):
                values=metrics[(rate,'torso','',name)][0]
                for stat,v in [('mean',values.mean()),('p95',np.percentile(values,95)),('max',values.max())]:
                    if v>original[name][stat]+1e-6:return False
        return True

    @staticmethod
    def contact_ceiling(p,height):
        return max(p.floor+.00035,float(height)-POLICY['foot_contact_step_m'])

    def elastic_foot_qp(self,p,P,g,A,lower,upper):
        """Soften contact descent only during search; all original final gates remain hard."""
        import osqp
        contact=np.flatnonzero(np.isneginf(lower)&np.isfinite(upper))
        count=len(contact)
        if not count:raise ValueError('No contact descent rows in elastic contact proposal')
        slack=sparse.coo_matrix((-np.ones(count),(contact,np.arange(count))),shape=(A.shape[0],count)).tocsc()
        matrix=sparse.vstack([sparse.hstack([A,slack]),sparse.hstack([sparse.csc_matrix((count,p.N)),sparse.eye(count)])],format='csc')
        objective=sparse.block_diag([P,sparse.eye(count)*POLICY['contact_slack_weight']],format='csc')
        solver=osqp.OSQP();solver.setup(P=sparse.triu(objective).tocsc(),q=np.r_[g,np.zeros(count)],A=matrix,
            l=np.r_[lower,np.zeros(count)],u=np.r_[upper,np.full(count,np.inf)],
            eps_abs=1e-7,eps_rel=1e-7,max_iter=10000,polishing=True,verbose=False)
        solution=solver.solve();status=str(solution.info.status)
        valid=solution.x is not None and np.isfinite(solution.x).all() and status in ('solved','solved inaccurate','maximum iterations reached')
        info=dict(status=status,iterations=int(solution.info.iter),primal_residual=float(solution.info.prim_res),
            variables=len(g)+count,constraints=matrix.shape[0],elastic_contact_rows=count,
            contact_slack_max_mm=float(np.max(solution.x[p.N:])*5) if valid else None,
            final_contact_gates_unchanged=True)
        return solution.x[:p.N].reshape(p.n,p.d)*p.scales if valid else None,info

    def finish_lower(self,p,x):
        before=p.measure(x)
        if before['grounded_target_height_max_mm']<=.45:return x
        trial=self.vertical_project(p,x)
        if trial is None:return x
        after=p.measure(trial)
        accepted=after['feasible'] and after['grounded_target_height_max_mm']<before['grounded_target_height_max_mm']-1e-5
        self.record(dict(stage='lower_closure',applied=accepted,before=finite(before),after=finite(after),final_admission_required=True))
        return trial if accepted else x

    def elastic_body_qp(self,p,P,g,A,lower,upper):
        import osqp
        rows=np.flatnonzero(np.isneginf(lower)&np.isfinite(upper));count=len(rows)
        slack=sparse.coo_matrix((-np.ones(count),(rows,np.arange(count))),shape=(A.shape[0],count)).tocsc()
        matrix=sparse.vstack([sparse.hstack([A,slack]),sparse.hstack([sparse.csc_matrix((count,p.N)),sparse.eye(count)])],format='csc')
        objective=sparse.block_diag([P,sparse.eye(count)*1e8],format='csc')
        solver=osqp.OSQP();solver.setup(P=sparse.triu(objective).tocsc(),q=np.r_[g,np.zeros(count)],A=matrix,
            l=np.r_[lower,np.zeros(count)],u=np.r_[upper,np.full(count,np.inf)],eps_abs=1e-7,eps_rel=1e-7,max_iter=10000,polishing=True,verbose=False)
        solution=solver.solve();status=str(solution.info.status)
        valid=solution.x is not None and np.isfinite(solution.x).all() and status in ('solved','solved inaccurate','maximum iterations reached')
        info=dict(status=status,iterations=int(solution.info.iter),primal_residual=float(solution.info.prim_res),elastic_stat_rows=count,
            max_stat_slack=float(np.max(solution.x[p.N:])) if valid else None,final_admission_unchanged=True)
        return solution.x[:p.N].reshape(p.n,p.dim)*p.scale if valid else None,info

    def preserve_repair_columns(self,p,trial):
        """A one-arm repair leaves the other arm byte-exact at its seed."""
        if getattr(self,'repair_fixed',None) is None:return trial
        columns,seed=self.repair_fixed
        result=trial.copy();result[:,columns]=seed[:,columns]
        return result

    @staticmethod
    def repair_rank(candidate):
        _,score,_,mode=candidate
        return (score['primary_regressions'],mode=='both',score['primary_cost'])

    def body(self,p):
        score,metrics,_=p.evaluate(p.x0,False)
        if score['physical'] and self.torso_nonregression(p,metrics):
            self.record(dict(stage='body',method='preserve_canonical_nonregressing_torso',applied=False,
                before=finite(score),after=finite(score),original_source_comparison=True,final_admission_required=True))
            return p.x0.copy(),dict(method='preserve_canonical_nonregressing_torso',before=score,after=score,history=[])
        if hasattr(self.batch,'_hard_original_body'):
            started=time.monotonic()
            x=p.x0.copy();before,_=p.source_measure(x);current=before;history=[]
            def violation(metrics):
                total=0.
                for rate in (120,50):
                    for name in ('torso_orientation_error_deg','torso_lateral_error_deg'):
                        baseline=p.original['audit_h2'][str(rate)][name];v=metrics[(rate,'torso','',name)][0]
                        for stat,value in [('mean',v.mean()),('p95',np.percentile(v,95)),('max',v.max())]:
                            total+=max(0.,(float(value)-baseline[stat]-1e-6)/max(baseline[stat],.1))**2
                return total
            for iteration in range(24):
                current,metrics=p.source_measure(x,True);old=violation(metrics)
                if old<=1e-18:break
                delta,info=p.source_qp(x,metrics)
                if delta is None:break
                winner=None
                for alpha in (1.,.5,.25,.125,.0625,.03125):
                    trial=np.clip(x+alpha*delta,p.lo,p.hi);score,mm=p.source_measure(trial,False);v=violation(mm)
                    if score['physical'] and (v<old-1e-15 or v<=1e-18):winner=(trial,score,v);break
                history.append(dict(iteration=iteration+1,accepted=winner is not None,violation=old,qp=info))
                if winner is None:break
                x,current,_=winner
            receipt=dict(method='canonical_torso_violation_restoration',before=before,after=current,history=history)
            self.record(dict(stage='body',method='joint_torso_arm_residual_correction',applied=not np.array_equal(x,p.x0),
                before=finite(score),after=finite(receipt['after']),seconds=time.monotonic()-started,final_admission_required=True))
            return x,receipt
        checks=[c for c in p.seed_report['checks'] if c['section']=='torso']
        if checks and all(c['passed'] for c in checks):
            score,_=p.source_measure(p.x0)
            return p.x0.copy(),dict(method='reuse_qualified_source_torso',before=score,after=score,history=[])
        import fast_primitives
        started=time.monotonic();x,receipt=fast_primitives.torso(p,self.batch)
        self.record(dict(stage='body',method=receipt['method'],applied=not np.array_equal(x,p.x0),seconds=time.monotonic()-started,
            before=receipt['before'],after=receipt['after'],analytic_model_fk_error_rad=receipt['analytic_model_fk_error_rad'],final_admission_required=True))
        return x,receipt

    @staticmethod
    def angular_mean(metrics):
        total=0.
        for rate in (120,50):
            for side in ('Left','Right'):
                for name in ('upper_direction_error_deg','forearm_direction_error_deg','elbow_flex_error_deg'):
                    total+=float(metrics[(rate,'arms',side,name)][0].mean())
            for name in ('torso_orientation_error_deg','torso_lateral_error_deg'):
                total+=float(metrics[(rate,'torso','',name)][0].mean())
        return total

    def upper(self,p,initial,current,check,admissible,restore):
        if self.disabled:return initial,current
        began=time.monotonic()
        _,initial_metrics=check(initial)
        _,_,observed=p.evaluate(initial,False)
        torso=[v for v in observed if v['check']['section']=='torso']
        arm_primary=[v for v in observed if v['check']['section']=='arms' and v['check']['metric']!='wrist_orientation_error_deg']
        if (arm_primary and all(v['over']<=0 for v in arm_primary) and current['physical']
            and not current['primary_regressions'] and current['wrist_max_deg']<=15
            and self.torso_nonregression(p,initial_metrics)):
            self.goal_ready=True
            self.record(dict(stage='upper',method='independently_qualified_arms_preserved',applied=False,
                before=finite(current),after=finite(current),arm_g1_checks_passed=True,
                torso_original_nonregression_pass=True,final_admission_required=True))
            return initial,current
        small=(current['physical'] and not current['primary_regressions'] and current['wrist_max_deg']<=15
               and self.torso_nonregression(p,initial_metrics))
        for rate in (120,50,600):
            for side in ('Left','Right'):
                for name in ('upper_direction_error_deg','forearm_direction_error_deg','elbow_flex_error_deg'):
                    small=small and float(initial_metrics[(rate,'arms',side,name)][0].max())<=1.
                small=small and float(initial_metrics[(rate,'arms',side,'normalized_hand_error')][0].max())<=.005
        if small:
            self.goal_ready=True
            self.record(dict(stage='upper',method='reuse_already_accurate_source_arms',applied=False,fast_goal_ready=True,seconds=time.monotonic()-began,
                before=finite(current),after=finite(current),angular_max_limit_deg=1.,normalized_hand_error_limit=.005,
                torso_canonical_nonregression_pass=True,original_primary_envelopes_unchanged=True,final_admission_required=True))
            return initial,current
        q=p.assemble(initial);sides=['Left','Right']
        bodies=[p.torso]+[p.arm_bodies[s][-1] for s in sides]
        _,rotation=self.batch.U.body_poses(p.m,q,bodies)
        proposals={name:np.zeros_like(initial) for name in ('retrieved_residual','retrieved_pose')};all_donors={}
        for si,side in enumerate(sides):
            js=np.asarray(self.bank.meta['arm_indices'][side.lower()]);local=np.flatnonzero(np.isin(p.js,js))
            if len(local)!=7:continue
            feature=arm_features(q,js,rotation[:,0],p.refs[120]['arms'][side]['dirs'],rotation[:,si+1],p.refs[120]['arms'][side]['wrist'])
            for kind,absolute,k in [('retrieved_residual',False,8),('retrieved_pose',True,1)]:
                delta,distance,owner=self.bank.query(side.lower(),feature,self.key,k=k,absolute=absolute)
                if absolute:delta-=initial[:,local]
                good=np.isfinite(distance)&(distance<=7.)
                delta=gaussian_filter1d(delta,sigma=1.5,axis=0,mode='nearest');delta[~good]=0.
                proposals[kind][:,local]=delta;all_donors[side+'_'+kind]=self.donors(owner)
        import fast_primitives
        try:proposals['analytic_wrist_rotation']=fast_primitives.wrists(p,self.batch)-initial
        except ValueError:pass
        try:proposals['native_residual_fit']=fast_primitives.arms(p,self.batch,initial,POLICY['native_residual_steps'])-initial
        except (ValueError,np.linalg.LinAlgError):pass
        _,baseline_metrics=check(initial)
        winner=initial;best=current;best_metrics=baseline_metrics;trials=[];repair_candidates=[];winning_method='none'
        for kind,proposed in proposals.items():
            for mode in ('both','left','right'):
                delta=proposed.copy()
                if mode!='both':
                    chosen=np.array([p.names[j].startswith(mode+'_') for j in p.js]);delta[:,~chosen]=0.
                for amount in (1.,.5,.25,.125):
                    trial=project_steps(initial+amount*delta,p.lo,p.hi,p.steps)
                    score,metrics=check(trial)
                    improves=(score['primary_cost']<best['primary_cost']-1e-5 or (best['wrist_max_deg']>15 and score['wrist_max_deg']<best['wrist_max_deg']-1e-4))
                    accepted=(admissible(score,metrics,'primary') and score['wrist_max_deg']<=max(15.+1e-6,best['wrist_max_deg']-1e-4) and improves)
                    trials.append(dict(kind=kind,mode=mode,amount=amount,accepted=accepted,score=finite(score)))
                    if accepted:winner,best,best_metrics,winning_method=trial,score,metrics,kind;break
                    if (kind!='analytic_wrist_rotation' and score['physical'] and
                        score['primary_cost']<current['primary_cost']*(1-POLICY['repair_min_relative_cost_gain']) and
                        score['wrist_max_deg']<=15 and 0<score['primary_regressions']<=POLICY['repair_max_statistical_regressions']):
                        repair_candidates.append((trial,score,metrics,mode))
        # One original constraint restoration is permitted; no full cold SQP retry.
        ordered=sorted(repair_candidates,key=self.repair_rank)
        candidates=[];used_modes=set()
        for candidate in ordered:
            if candidate[3] not in used_modes:candidates.append(candidate);used_modes.add(candidate[3])
        candidates.extend(c for c in ordered if all(c is not existing for existing in candidates))
        for candidate in candidates[:POLICY['repair_candidates']] if np.array_equal(winner,initial) else []:
            trial,score,metrics,mode=candidate
            before_repair=finite(score);repair_started=time.monotonic()
            inactive=np.array([not p.names[j].startswith(mode+'_') for j in p.js]) if mode!='both' else np.zeros(len(p.js),dtype=bool)
            self.repair_fixed=(inactive,initial)
            try:repaired,score,metrics,repairs=restore(trial,score,metrics,'primary')
            finally:self.repair_fixed=None
            accepted=admissible(score,metrics,'primary') and score['primary_cost']<current['primary_cost']-1e-5
            trials.append(dict(kind='original_constraint_projection',mode=mode,accepted=accepted,repairs=repairs,
                before_repair=before_repair,score=finite(score),seconds=time.monotonic()-repair_started,
                candidates_considered=len(repair_candidates),unchanged_arm_preserved_exact=True))
            if accepted:winner,best,best_metrics,winning_method=repaired,score,metrics,'retrieved_constraint_projection'
            if accepted:break
        _,_,observed=p.evaluate(initial,False)
        primary=[v for v in observed if v['check']['metric']!='wrist_orientation_error_deg']
        source_qualified=(bool(primary) and all(v['over']<=0 for v in primary) and current['physical']
            and not current['primary_regressions'] and current['wrist_max_deg']<=15
            and self.torso_nonregression(p,baseline_metrics))
        relative=(current['primary_cost']-best['primary_cost'])/max(abs(current['primary_cost']),1.)
        mean_gain=self.angular_mean(baseline_metrics)-self.angular_mean(best_metrics)
        source_progress=(relative>=.05 and mean_gain>=1.) or (current['wrist_max_deg']>15 and best['wrist_max_deg']<=15 and current['wrist_max_deg']-best['wrist_max_deg']>=1.)
        reusable_improvement=(best['primary_cost']<current['primary_cost']-1e-5 or
                              (current['wrist_max_deg']>15 and best['wrist_max_deg']<=15))
        self.goal_ready=bool(source_qualified or (not np.array_equal(winner,initial) and best['physical'] and not best['primary_regressions'] and best['wrist_max_deg']<=15 and reusable_improvement))
        self.record(dict(stage='upper',method=winning_method,applied=not np.array_equal(winner,initial),seconds=time.monotonic()-began,
            before=finite(current),after=finite(best),donors=all_donors,trials=trials,fast_goal_ready=self.goal_ready,
            primary_relative_cost_gain=relative,primary_mean_angular_gain_deg=mean_gain,
            original_primary_envelopes_unchanged=True,legacy_fast_gain_threshold_met=source_progress,
            minimum_progress_gate_removed=True,final_admission_required=True))
        if not self.goal_ready:
            self.fast=False
            self.record(dict(stage='upper_residual',method='original_canonical_sqp',iterations=POLICY['residual_sqp_steps'],final_admission_required=True))
        return winner,best
