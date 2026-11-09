"""Optimizer-only convergence controls; original independent admission is intact."""
import ast
import inspect
import math
import textwrap
import time
from pathlib import Path
import numpy as np

POLICY=dict(format='adaptive_geometry_progress_v1',admission_changed=False,
    wall_timeout_seconds=None,campaign_timeout_seconds=None,
    foot_contact_stop_mm=.45,plateau_steps=3,
    plateau_joint_step_deg=.01,plateau_root_step_mm=.05,
    plateau_relative_cost_gain=1e-4,plateau_contact_gain_mm=1e-4,
    fixed_reference_stop_fraction=.9,
    note='Stage feasibility triggers original final admission, never admission by itself. Stagnant failures remain review.')

def finite(value):
    if isinstance(value,dict):return {k:finite(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [finite(v) for v in value]
    if isinstance(value,(float,np.floating)) and not math.isfinite(value):return None
    if isinstance(value,np.generic):return value.item()
    return value

def stalled(old,new,old_cost,new_cost,foot=False,contact_gain=0):
    diff=np.abs(np.asarray(new)-np.asarray(old))
    joints=diff[:,3:] if foot else diff
    joint_step=float(np.rad2deg(joints).max(initial=0))
    root_step=float(np.linalg.norm(diff[:,:3],axis=1).max(initial=0)*1000) if foot else 0.
    relative=max(0.,float(old_cost)-float(new_cost))/max(abs(float(old_cost)),1.)
    tiny=joint_step<=POLICY['plateau_joint_step_deg'] and root_step<=POLICY['plateau_root_step_mm'] and relative<=POLICY['plateau_relative_cost_gain']
    if foot:tiny=tiny and contact_gain<=POLICY['plateau_contact_gain_mm']
    return tiny,dict(max_joint_step_deg=joint_step,max_root_step_mm=root_step,relative_cost_gain=relative,contact_gain_mm=contact_gain)

def primary_ready(problem,x,current):
    if not current.get('physical') or current.get('primary_regressions',0):return False
    _,_,observed=problem.evaluate(x,False)
    primary=[o for o in observed if o['check']['metric']!='wrist_orientation_error_deg']
    return bool(primary) and all(o['over']<=0 for o in primary) and current.get('wrist_max_deg',0)<=15.

def fixed_ready(problem,x):
    score,metrics=problem.source_measure(x,False)
    if not score['physical']:return False
    fraction=POLICY['fixed_reference_stop_fraction']
    for rate in (120,50,600):
        for body in range(7):
            v=np.stack([metrics[(rate,'fixed','',f'position_{body}_{axis}')][0] for axis in range(3)],1)
            if float(np.linalg.norm(v,axis=1).max(initial=0))>fraction:return False
        for body in (2,5,6):
            v=np.stack([metrics[(rate,'fixed','',f'rotation_{body}_{axis}')][0] for axis in range(3)],1)
            if float(np.linalg.norm(v,axis=1).max(initial=0))>fraction:return False
        for side in ('Left','Right'):
            if metrics[(rate,'arms',side,'wrist_orientation_error_deg')][0].max()>15.:return False
    # A geometrically close refit can still violate the original1e-6 raw envelopes.
    for rate in (120,50):
        original=problem.original['audit_h2'][str(rate)]
        records=[]
        for side in ('Left','Right'):
            for name in ('upper_direction_error_deg','forearm_direction_error_deg','elbow_flex_error_deg','normalized_hand_error'):
                records.append((metrics[(rate,'arms',side,name)][0],original['arms'][side][name]))
        for name in ('torso_orientation_error_deg','torso_lateral_error_deg'):
            records.append((metrics[(rate,'torso','',name)][0],original[name]))
        for values,baseline in records:
            for stat,value in [('mean',float(values.mean())),('p95',float(np.percentile(values,95))),('max',float(values.max()))]:
                if value>baseline[stat]+1e-6:return False
    return True

class Controller:
    def __init__(self,batch,out,key,save):
        self.batch,self.out,self.key,self.save=batch,Path(out),key,save
        self.states={};self.events=[];self.attempt=0;self.disable_feasible=False

    def begin(self,attempt,disable_feasible=False):
        self.states={};self.attempt=attempt;self.disable_feasible=disable_feasible

    def tick(self,stage,problem,x,current,iteration):
        token=(stage,id(problem));state=self.states.setdefault(token,dict(previous=x.copy(),cost=current['cost'] if 'cost' in current else current.get('primary_cost',0),contact=current.get('grounded_target_height_max_mm'),streak=0))
        cost=current.get('cost',current.get('primary_cost',0));previous_contact=state.get('contact');contact=current.get('grounded_target_height_max_mm')
        contact_gain=(previous_contact-contact) if previous_contact is not None and contact is not None and np.isfinite(previous_contact) and np.isfinite(contact) else 0.
        tiny,delta=stalled(state['previous'],x,state['cost'],cost,stage=='feet',contact_gain)
        state['streak']=state['streak']+1 if iteration>0 and tiny else 0
        reason=None
        if not self.disable_feasible:
            if stage=='feet' and current['feasible'] and (contact == -np.inf or (np.isfinite(contact) and contact<=POLICY['foot_contact_stop_mm'])):reason='stage_feasible_pending_full_admission'
            elif stage=='upper' and primary_ready(problem,x,current):reason='primary_g1_ready_pending_full_admission'
            elif stage=='refit' and fixed_ready(problem,x):reason='fixed_goal_ready_pending_full_admission'
        if reason is None and state['streak']>=POLICY['plateau_steps']:reason='negligible_change_plateau'
        state.update(previous=x.copy(),cost=cost,contact=contact)
        event=dict(attempt=self.attempt,stage=stage,iteration=iteration,stop_reason=reason,score=finite(current),delta=delta,plateau_steps=state['streak'],timestamp_unix=time.time())
        self.save(self.out/'jobs'/self.key/'adaptive_progress.json',event)
        if reason:
            self.events.append(event);self.save(self.out/'jobs'/self.key/'adaptive_stops.json',self.events)
        return reason

def rewrite(fn,stage,namespace):
    """Insert only loop exit hooks into the unchanged numerical optimizer."""
    tree=ast.parse(textwrap.dedent(inspect.getsource(fn)));node=tree.body[0]
    # Recognize initialization and completed-iteration insertion points explicitly.
    if stage=='feet':
        loop=next(n for n in node.body if isinstance(n,ast.For) and ast.unparse(n.target)=='iteration')
        loop.body.extend(ast.parse("if _adaptive.tick('feet', p, x, current, iteration+1):\n    break").body)
        if '_reuse' in namespace:
            before=next(n for n in node.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='before' for t in n.targets))
            index_before=node.body.index(before)
            node.body[index_before+1:index_before+1]=ast.parse('x = _reuse.lower(p, x)\ncurrent = p.measure(x)').body
        index=node.body.index(loop)
        # Source loop can still be skipped for an already feasible seed.
        node.body[index:index+1]=ast.parse("_stage_stop = _adaptive.tick('feet', p, x, current, 0)").body+[ast.If(test=ast.UnaryOp(op=ast.Not(),operand=ast.Name(id='_stage_stop',ctx=ast.Load())),body=[loop],orelse=[])]
        if '_reuse' in namespace:
            pos=index+1
            node.body[pos+1:pos+1]=ast.parse('x = _reuse.finish_lower(p, x)\ncurrent = p.measure(x)').body
    elif stage=='upper':
        outer=next(n for n in node.body if isinstance(n,ast.For) and ast.unparse(n.target)=='phase')
        loop=next(n for n in outer.body if isinstance(n,ast.For))
        if '_reuse' in namespace:
            loop.iter=ast.parse('range(6)',mode='eval').body
            stats_index=next(i for i,n in enumerate(node.body) if isinstance(n,ast.FunctionDef) and n.name=='primary_records')
            node.body[stats_index:stats_index]=ast.parse('canonical_stats = _reuse.original_stats(self, canonical_stats)').body
            index_outer=node.body.index(outer)
            node.body[index_outer:index_outer]=ast.parse('x, current = _reuse.upper(self, x, current, check, admissible, restore)').body
            restoration=next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name=='restore')
            constraints=next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name=='linear_constraints')
            removed=0
            for candidate in ast.walk(constraints):
                if not isinstance(candidate,ast.For) or ast.unparse(candidate.target)!='(key, cap)':continue
                retained=[]
                for statement in candidate.body:
                    if isinstance(statement,ast.For) and ast.unparse(statement.target)=='i':removed+=1
                    else:retained.append(statement)
                candidate.body=retained
            assert removed==1,'Original per-frame search envelope structure changed'
            for candidate in ast.walk(restoration):
                if isinstance(candidate,ast.For) and isinstance(candidate.target,ast.Name):
                    if candidate.target.id=='repair_iteration':candidate.iter=ast.parse('range(6)',mode='eval').body
                    if candidate.target.id=='alpha':candidate.iter=ast.parse('(1.,.5,.25,.125,.0625,.03125)',mode='eval').body
            inserted=0
            for nested in ast.walk(restoration):
                if not isinstance(nested,ast.For):continue
                for index,statement in list(enumerate(nested.body)):
                    if isinstance(statement,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='trial' for t in statement.targets):
                        nested.body[index+1:index+1]=ast.parse('trial = _reuse.preserve_repair_columns(self, trial)').body
                        inserted+=1
            assert inserted==1,'Original restoration trial structure changed'
        loop.body.insert(0,ast.parse("if _adaptive.tick('upper', self, x, current, iteration):\n    break").body[0])
        if '_reuse' in namespace:
            loop.body.insert(0,ast.parse('if _reuse.fast and _reuse.goal_ready:\n    break').body[0])
    else:
        if '_reuse' in namespace:
            node.body.insert(0,ast.parse("if _reuse.fast and not hasattr(self, 'fixed_goal'):\n    return _reuse.body(self)").body[0])
        loop=next(n for n in node.body if isinstance(n,ast.For) and ast.unparse(n.target)=='iteration')
        if '_reuse' in namespace:
            for nested in ast.walk(loop):
                if not isinstance(nested,ast.For):continue
                for index,statement in list(enumerate(nested.body)):
                    if isinstance(statement,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='trial' for t in statement.targets):
                        nested.body[index+1:index+1]=ast.parse("if hasattr(self, 'fixed_goal'):\n    trial = _reuse.preserve_refit(self, trial)").body
        # Body reconstruction retains its objective; only fixed-goal refit uses geometry-ready exit.
        loop.body.insert(0,ast.parse("if hasattr(self, 'fixed_goal'):\n    if _adaptive.tick('refit', self, x, current, iteration):\n        break\nelse:\n    if _adaptive.tick('body', self, x, current, iteration):\n        break").body[0])
    ast.fix_missing_locations(tree)
    local={};exec(compile(tree,'<adaptive_'+stage+'>','exec'),namespace,local)
    return local[node.name]

def install(batch,out,key,save,reuse=None):
    controller=Controller(batch,out,key,save)
    # No changes to run_motion, audits, acceptance predicates, QP objectives or constraints.
    namespace=batch.__dict__.copy();namespace['_adaptive']=controller
    if reuse is not None:namespace['_reuse']=reuse
    foot=rewrite(batch.foot_project,'feet',namespace)
    def clean_foot(*args,**kwargs):
        q,receipt=foot(*args,**kwargs)
        return q,finite(receipt)
    batch.foot_project=clean_foot
    if reuse is not None:
        tree=ast.parse(textwrap.dedent(inspect.getsource(batch.F.SourceFootProblem.qp)))
        replaced=0
        for node in ast.walk(tree):
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='contact_ceiling' for t in node.targets):
                node.value=ast.parse('_reuse.contact_ceiling(self, height[side,corner])',mode='eval').body;replaced+=1
        assert replaced==1,'Original contact step structure changed'
        body=tree.body[0].body
        index=next(i for i,n in enumerate(body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='A' for t in n.targets))
        body[index+1:]=ast.parse('return _reuse.elastic_foot_qp(self,P,g,A,np.array(lo),np.array(hi))').body
        ast.fix_missing_locations(tree)
        scope=batch.F.__dict__.copy();scope['_reuse']=reuse;local={}
        exec(compile(tree,'<adaptive_contact_step>','exec'),scope,local)
        batch.F.SourceFootProblem.qp=local['qp']
    namespace=batch.U.__dict__.copy();namespace['_adaptive']=controller
    if reuse is not None:namespace['_reuse']=reuse
    upper=rewrite(batch.U.Problem.anatomical_project,'upper',namespace)
    refit=rewrite(batch.U.Problem.reconstruct,'refit',namespace)
    def clean_upper(*args,**kwargs):
        q,receipt=upper(*args,**kwargs)
        return q,finite(receipt)
    def clean_refit(*args,**kwargs):
        q,receipt=refit(*args,**kwargs)
        return q,finite(receipt)
    batch.U.Problem.anatomical_project=clean_upper
    batch.U.Problem.reconstruct=clean_refit
    # Torso repair uses the same immutable statistical envelopes as final review.
    tree=ast.parse(textwrap.dedent(inspect.getsource(batch.U.Problem.source_qp)))
    body=tree.body[0].body
    index=next(i for i,n in enumerate(body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='D' for t in n.targets))
    insert='''
if not hasattr(self, 'fixed_goal'):
    for rate in (120,50):
        for section,side,name in [('torso','','torso_orientation_error_deg'),('torso','','torso_lateral_error_deg')]:
            values,grad,left,right,alpha=metrics[(rate,section,side,name)]
            baseline=self.original['audit_h2'][str(rate)][name]
            order=np.argsort(values);position=.95*(len(values)-1);ilo=int(np.floor(position));ihi=int(np.ceil(position));fraction=position-ilo
            for stat,indices,weights in [('mean',np.arange(len(values)),np.full(len(values),1/len(values))),('p95',np.array([order[ilo],order[ihi]]),np.array([1-fraction,fraction])),('max',np.array([values.argmax()]),np.array([1.]))]:
                entries=[]
                scale=max(baseline['max'],.1)
                for i,w in zip(indices,weights):
                    for frame,mix in ((left[i],1-alpha[i]),(right[i],alpha[i])):
                        entries.extend((int(frame*self.dim+j),float(v*self.scale*mix*w/scale)) for j,v in enumerate(grad[i]) if abs(v)>1e-14 and mix>1e-10)
                constraint(entries,-np.inf,(baseline[stat]-float(values[indices]@weights)-2e-7)/scale)
'''
    body[index:index]=ast.parse(textwrap.dedent(insert)).body
    body=tree.body[0].body
    index=next(i for i,n in enumerate(body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='A' for t in n.targets))
    body[index+1:]=ast.parse("if not hasattr(self, 'fixed_goal'):\n    return _reuse.elastic_body_qp(self,P,g,A,np.array(lower),np.array(upper))\n" ).body+body[index+1:]
    scope=batch.U.__dict__.copy();scope['_reuse']=reuse
    ast.fix_missing_locations(tree);local={};exec(compile(tree,'<canonical_torso_qp>','exec'),scope,local)
    batch.U.Problem.source_qp=local['source_qp']
    return controller
