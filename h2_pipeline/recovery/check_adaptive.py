"""Convergence never promotes a failed motion; tests cover competing signals."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import numpy as np
if __package__:
    from .adaptive_stopping import Controller,stalled,finite
else:
    from adaptive_stopping import Controller,stalled,finite

def main():
    a=np.zeros((4,15));b=a.copy();b[:,3:]=np.deg2rad(.005)
    assert stalled(a,b,100,99.999,foot=True)[0]
    assert not stalled(a,b,100,99.999,foot=True,contact_gain=.01)[0]
    assert not stalled(a,b,100,90,foot=True)[0]
    b[:,0]=.001
    assert not stalled(a,b,100,99.999,foot=True)[0]
    saved=[]
    c=Controller(None,Path('/unused'),'test',lambda p,v:saved.append((p,v)))
    c.begin(1)
    score=dict(cost=100.,feasible=True,grounded_target_height_max_mm=.7)
    assert c.tick('feet',None,a,score,0) is None
    for i in (1,2):assert c.tick('feet',None,a,score,i) is None
    assert c.tick('feet',None,a,score,3)=='negligible_change_plateau'
    c.begin(2)
    good=dict(score,grounded_target_height_max_mm=.4)
    assert c.tick('feet',None,a,good,0)=='stage_feasible_pending_full_admission'
    c.begin(3)
    assert c.tick('feet',None,a,dict(good,feasible=False),0) is None
    c.begin(4)
    assert c.tick('feet',None,a,dict(good,grounded_target_height_max_mm=float('inf')),0) is None
    c.begin(5,disable_feasible=True)
    assert c.tick('feet',None,a,good,0) is None
    json.dumps(finite(dict(v=float('-inf'),a=[np.float64(1.)])),allow_nan=False)
    assert all('accepted' not in str(v.get('stop_reason','')) for p,v in saved if isinstance(v,dict))
    print('PASS: useful contact progress retained; 3-step plateau; feasible exits require original final admission; nonfinite metadata; fallback controls')

if __name__=='__main__':main()
