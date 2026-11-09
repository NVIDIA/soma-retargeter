"""Only already-accurate physical arms with G1-qualified torso may skip search."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
if __package__:
    from .retrieved_start import Reuse
else:
    from retrieved_start import Reuse

def main():
    def exercise(angle=.02,hand=.0001,torso_over=0.,physical=True):
        x=np.zeros((3,14));metrics={}
        for rate in (120,50,600):
            for side in ('Left','Right'):
                for name in ('upper_direction_error_deg','forearm_direction_error_deg','elbow_flex_error_deg'):
                    metrics[(rate,'arms',side,name)]=(np.full(3,angle),)
                metrics[(rate,'arms',side,'normalized_hand_error')]=(np.full(3,hand),)
            for name in ('torso_orientation_error_deg','torso_lateral_error_deg'):
                metrics[(rate,'torso','',name)]=(np.full(3,.01+torso_over),)
        current=dict(physical=physical,wrist_max_deg=.01,primary_regressions=0)
        def not_shortcut(*args):raise RuntimeError('NOT_SHORTCUT')
        original={'audit_h2':{str(rate):{name:{s:.01 for s in ('mean','p95','max')} for name in ('torso_orientation_error_deg','torso_lateral_error_deg')} for rate in (120,50)}}
        p=SimpleNamespace(original=original,evaluate=lambda *args:(None,None,[dict(check=dict(section='torso'),over=torso_over)]),
            assemble=lambda x:x,torso=0,arm_bodies={'Left':[0,1,2],'Right':[3,4,5]},m=None)
        reuse=Reuse.__new__(Reuse);reuse.disabled=False;reuse.fast=True;reuse.goal_ready=False;reuse.events=[]
        reuse.batch=SimpleNamespace(U=SimpleNamespace(body_poses=not_shortcut));reuse.out=Path('/unused');reuse.key='test';reuse.save=lambda *args:None
        try:
            actual,score=reuse.upper(p,x,current,lambda x:(current,metrics),None,None)
            assert np.array_equal(actual,x) and score==current and reuse.goal_ready
            return True
        except RuntimeError as e:
            assert str(e)=='NOT_SHORTCUT';return False
    assert exercise()
    assert not exercise(angle=1.01)
    assert not exercise(hand=.0051)
    assert not exercise(torso_over=.01)
    assert not exercise(physical=False)
    print('PASS: accurate-source shortcut preserves trajectory and requires angular/hand/physical and immutable original torso envelopes')

if __name__=='__main__':main()
