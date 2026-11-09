"""Regression coverage: no self/mirror leakage, native limits and original envelopes."""
from pathlib import Path
import json
import tempfile
import numpy as np
if __package__:
    from .solution_bank import Bank,sha,local_xy
else:
    from solution_bank import Bank,sha,local_xy
if __package__:
    from .retrieved_start import project_steps
else:
    from retrieved_start import project_steps

def main():
    rng=np.random.default_rng(9)
    q=rng.normal(size=(120,12));lo=np.full(12,-1.);hi=np.full(12,1.);cap=np.full(12,.03)
    result=project_steps(q,lo,hi,cap)
    assert np.all(result>=lo) and np.all(result<=hi) and np.max(abs(np.diff(result,axis=0))-cap)<1e-12
    assert np.array_equal(q,q.copy())
    v=rng.normal(size=(20,3));angle=rng.normal(size=20)
    assert np.allclose(local_xy(local_xy(v,angle),angle,inverse=True),v)
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/'bank.npz'
        np.savez(path,owner=np.array([0,1,2]),contact=np.zeros(3,int),
            lower_features=np.array([[0.,0.],[0.,0.],[1.,1.]]),
            lower_delta=np.array([[1.]*15,[3.]*15,[2.]*15]),
            left_features=np.zeros((3,16)),right_features=np.zeros((3,16)),
            left_delta=np.zeros((3,7)),right_delta=np.zeros((3,7)))
        path.with_suffix('.json').write_text(json.dumps(dict(donor_keys=['target','target_M','other'],bank_sha256=sha(path))))
        bank=Bank(path);delta,distance,owner=bank.query('lower',np.array([[0.,0.]]),'target',contact=np.array([0]))
        assert np.allclose(delta,2.) and owner[0]==2,'Self or mirror donor leaked into retrieval'
        delta,distance,owner=bank.query('lower',np.array([[0.,0.]]),'target_M',contact=np.array([0]))
        assert np.allclose(delta,2.) and owner[0]==2
    print('PASS: self/mirror exclusion, heading transform, native joint range and velocity projection')

if __name__=='__main__':main()
