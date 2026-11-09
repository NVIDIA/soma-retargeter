"""Small exact rotation primitives plus native curve projection; never admission."""
import numpy as np
from scipy import sparse
from scipy.spatial.transform import Rotation as R

def native_curve(values,lo,hi,step):
    import osqp
    n,dim=values.shape
    I=sparse.eye(n*dim,format='csc')
    D=sparse.kron(sparse.diags([-np.ones(n-1),np.ones(n-1)],[0,1],shape=(n-1,n)),sparse.eye(dim),format='csc')
    A=sparse.vstack([I,D],format='csc');caps=np.tile(step,n-1)
    solver=osqp.OSQP();solver.setup(P=I,q=-values.ravel(),A=A,
        l=np.r_[np.tile(lo,n),-caps],u=np.r_[np.tile(hi,n),caps],eps_abs=1e-9,eps_rel=1e-9,max_iter=6000,polishing=True,verbose=False)
    try:
        # A proposal need not launch an expensive fallback solve. It is independently checked later.
        solution=solver.solve_interior() if hasattr(solver,'solve_interior') else solver.solve()
    except Exception:
        if __package__:
            from .retrieved_start import project_steps
        else:
            from retrieved_start import project_steps
        return project_steps(values,lo,hi,step)
    if solution.x is None or not np.isfinite(solution.x).all():raise ValueError('No finite native-curve projection')
    result=solution.x.reshape(n,dim)
    if max(float((lo-result).max()),float((result-hi).max()))>1e-7 or float((abs(np.diff(result,axis=0))/step).max())>1.000001:raise ValueError('Native-curve projection failed physical constraints')
    return result

def torso(problem,batch):
    # Verify the actual model chain rather than assume an Euler convention.
    p=problem
    chain=[('waist_yaw_joint',[0,0,1]),('waist_roll_joint',[1,0,0]),('waist_pitch_joint',[0,1,0])]
    for name,axis in chain:
        j=p.m.joint(name).id;b=p.m.jnt_bodyid[j]
        assert np.array_equal(p.m.jnt_axis[j],axis) and np.array_equal(p.m.body_quat[b],[1,0,0,0])
    js=np.array([p.names.index(n) for n,a in chain]);local=np.array([np.flatnonzero(p.js==j)[0] for j in js])
    root=R.from_quat(p.q0[:,[4,5,6,3]])
    target=R.from_matrix(p.refs[120]['torso'])
    angles=(root.inv()*target).as_euler('ZXY')
    check=p.q0.copy();check[:,7+js]=angles
    _,actual=batch.U.body_poses(p.m,check,[p.torso])
    mismatch=float((R.from_matrix(actual[:,0]).inv()*target).magnitude().max())
    assert mismatch<1e-6,'Torso rotation primitive does not match this model'
    x=p.x0.copy();x[:,local]=native_curve(angles,p.lo[local],p.hi[local],p.steps[local])
    before,_=p.source_measure(p.x0);after,_=p.source_measure(x)
    assert after['physical']
    return x,dict(method='model_verified_analytic_torso_projection',before=before,after=after,history=[],
        source_goals='Unchanged mapped BVH torso rotations; cached upper correction follows',
        analytic_model_fk_error_rad=mismatch,no_training_admission=True)

def wrists(p,batch):
    q=p.q0.copy();proposal=p.x0.copy()
    for side in ('Left','Right'):
        prefix=side.lower();names=[prefix+'_wrist_'+s+'_joint' for s in ('roll','pitch','yaw')]
        js=np.array([p.names.index(n) for n in names]);local=np.array([np.flatnonzero(p.js==j)[0] for j in js])
        for name,axis in zip(names,([1,0,0],[0,1,0],[0,0,1])):
            j=p.m.joint(name).id;b=p.m.jnt_bodyid[j]
            assert np.array_equal(p.m.jnt_axis[j],axis) and np.array_equal(p.m.body_quat[b],[1,0,0,0])
        elbow=p.m.body(prefix+'_elbow_link').id
        _,rotation=batch.U.body_poses(p.m,q,[elbow])
        relative=R.from_matrix(rotation[:,0]).inv()*R.from_matrix(p.refs[120]['arms'][side]['wrist'])
        raw=relative.as_euler('XYZ')
        # Euler branches are proposals. Range, native speed and raw arm envelopes still decide.
        for axis in (0,2):raw[:,axis]+=2*np.pi*np.round((p.x0[:,local[axis]]-raw[:,axis])/(2*np.pi))
        proposal[:,local]=native_curve(raw,p.lo[local],p.hi[local],p.steps[local])
    return proposal

def arms(p,batch,initial,iterations=2):
    """Two batched native-frame residual steps; later checks decide admission."""
    x=initial.copy()
    for iteration in range(iterations):
        _,metrics,_=p.evaluate(x,True)
        candidate=x.copy()
        for side in ('Left','Right'):
            local=np.array([i for i,j in enumerate(p.js) if p.names[j].startswith(side.lower()+'_')])
            if len(local)!=7:continue
            keys=[f'direction_{i}_{j}' for i in range(2) for j in range(3)]+['flex_signed']
            rows=[metrics[(120,'aux',side,k)] for k in keys]
            residual=np.stack([r[0] for r in rows],axis=1)
            jac=np.stack([r[1][:,local] for r in rows],axis=1)
            factor=np.where(p.refs[120]['arms'][side]['straight'],2.,1.)
            residual[:,-1]*=factor;jac[:,-1]*=factor[:,None]
            wrist=[metrics[(120,'aux',side,f'wrist_log_{j}')] for j in range(3)]
            residual=np.concatenate([residual,np.stack([r[0] for r in wrist],axis=1)/15.],axis=1)
            jac=np.concatenate([jac,np.stack([r[1][:,local] for r in wrist],axis=1)/15.],axis=1)
            lhs=np.einsum('nri,nrj->nij',jac,jac)+np.eye(7)[None,:,:]
            rhs=-np.einsum('nri,nr->ni',jac,residual)
            delta=np.linalg.solve(lhs,rhs[...,None])[...,0]
            delta=np.clip(delta,-.12,.12)
            candidate[:,local]=native_curve(x[:,local]+delta,p.lo[local],p.hi[local],p.steps[local]*.9995)
        x=candidate
    return x
