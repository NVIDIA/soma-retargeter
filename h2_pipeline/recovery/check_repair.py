"""Regression coverage for repair ranking and exact preservation of the other arm."""
from pathlib import Path
from types import SimpleNamespace
import ast
import inspect
import textwrap
import numpy as np
if __package__:
    from .retrieved_start import Reuse
else:
    from retrieved_start import Reuse
if __package__:
    from .adaptive_stopping import rewrite
else:
    from adaptive_stopping import rewrite


def fixture(self):
    x=self.x0.copy();current={};canonical_stats={}
    def primary_records(metrics):return []
    def check(theta):
        assert np.array_equal(theta[:,1],self.x0[:,1])
        return {},{}
    def admissible(*args):return False
    def linear_constraints():
        rows=[]
        for key,cap in ():
            for i in range(0):rows.append(i)
            rows.append(cap)
        return rows
    def restore(theta,score,metrics,phase):
        for repair_iteration in range(1):
            for alpha in (1.,):
                trial=theta+1
                score,metrics=check(trial)
        return trial,score,metrics,[]
    for phase in ('primary',):
        for iteration in range(1):
            pass
    return x,current


def main():
    x=np.arange(12.).reshape(6,2)
    reuse=Reuse.__new__(Reuse);reuse.repair_fixed=(np.array([False,True]),x.copy())
    projected=reuse.preserve_repair_columns(None,x+1)
    assert np.array_equal(projected[:,1],x[:,1]) and np.array_equal(projected[:,0],x[:,0]+1)
    reuse.repair_fixed=None
    assert np.array_equal(reuse.preserve_repair_columns(None,x+1),x+1)
    a=(None,dict(primary_regressions=12,primary_cost=1),None,'both')
    b=(None,dict(primary_regressions=3,primary_cost=50),None,'left')
    assert min([a,b],key=Reuse.repair_rank) is b
    reuse.repair_fixed=(np.array([False,True]),x.copy())
    def upper(p,initial,current,check,admissible,restore):
        repaired,_,_,_=restore(initial,{}, {},'primary')
        assert np.array_equal(repaired[:,1],initial[:,1])
        return repaired,current
    reuse.upper=upper;reuse.fast=True;reuse.goal_ready=True
    modified=rewrite(fixture,'upper',dict(np=np,_reuse=reuse,_adaptive=SimpleNamespace(tick=lambda *args:False)))
    result,_=modified(SimpleNamespace(x0=x.copy(),original={'audit_h2':{}}))
    assert np.array_equal(result[:,1],x[:,1]) and np.array_equal(result[:,0],x[:,0]+1)
    print('PASS: repair ranking favors feasible-to-restore candidate; original restoration hook preserves unaffected arm before scoring')


if __name__=='__main__':main()
