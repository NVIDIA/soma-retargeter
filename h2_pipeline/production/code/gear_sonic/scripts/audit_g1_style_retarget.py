"""Same-source kinematic comparison: no force or policy-success claim."""
import argparse,csv,json,hashlib,os,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np,mujoco
from scipy.spatial.transform import Rotation as R,Slerp
ROOT=Path(__file__).resolve().parents[2]

def stats(x):
    x=np.asarray(x)
    return None if not x.size else dict(mean=float(x.mean()),p95=float(np.percentile(x,95)),max=float(x.max()))

def angle(a,b):
    return np.rad2deg(np.arccos(np.clip(np.sum(a*b,axis=-1),-1,1)))

def lateral(matrix,heading):
    up=matrix[...,2];left=np.stack([-np.sin(heading),np.cos(heading),np.zeros_like(heading)],axis=-1)
    return np.rad2deg(np.arctan2(np.sum(up*left,axis=-1),up[...,2]))

def yaw_error(a,b):
    ya=np.arctan2(a[:,1,0],a[:,0,0]);yb=np.arctan2(b[:,1,0],b[:,0,0])
    return np.rad2deg(np.arctan2(np.sin(ya-yb),np.cos(ya-yb)))

def evaluate(path,trace,robot,reference,fps=50.):
    m=mujoco.MjModel.from_xml_path(str(ROOT/'gear_sonic/data/assets/robot_description/mjcf'/('h2.xml' if robot=='h2' else 'g1_29dof_rev_1_0.xml')));d=mujoco.MjData(m)
    source=np.load(trace);raw=source['source_world'].astype(float);names=source['source_names'].tolist();native=float(source['fps'])
    scaler_path=reference.parent/'scaler.json' if robot=='h2' else Path(os.environ.get('SOMA_RETARGETER_ROOT',str(ROOT.parent/'soma-retargeter-h2')))/'soma_retargeter/configs/unitree_g1/soma_to_g1_scaler_config.json'
    scaler=json.loads(scaler_path.read_text())
    ref=np.load(reference);zero=ref['source'];zn=ref['source_names'].tolist();root_r=R.from_quat(zero[zn.index('Hips'),3:])*R.from_quat([.5,.5,.5,.5]);heading0=root_r.as_euler('xyz')[2];rz=R.from_euler('z',heading0)
    with path.open() as f:
        rd=csv.reader(f);header=[s.removesuffix('_dof') for s in next(rd)];data=np.array(list(rd),float)
    assert len(data)==len(raw) and np.array_equal(data[:,0],np.arange(len(data))) and np.isfinite(data).all()
    oldt=np.arange(len(data))/native;times=np.arange(int(np.floor(oldt[-1]*fps+1e-8))+1)/fps if len(data)>1 else np.array([0.])
    columns=[header.index(m.joint(i).name) for i in range(1,m.njnt)]
    q=np.zeros((len(times),m.nq));q[:,:3]=np.array([np.interp(times,oldt,data[:,i]) for i in (1,2,3)]).T
    rr=R.from_euler('xyz',data[:,4:7]);rr=Slerp(oldt,rr)(times) if len(data)>1 else rr
    q[:,3:7]=rr.as_quat()[:,[3,0,1,2]];q[:,7:]=np.array([np.interp(times,oldt,data[:,i]) for i in columns]).T
    human_pos=np.array([[np.interp(times,oldt,raw[:,j,k]) for k in range(3)] for j in range(len(names))]).transpose(2,0,1)
    human_rot={}
    for name in ('Hips','Chest','LeftFoot','RightFoot','LeftHand','RightHand'):
        sr=R.from_quat(raw[:,names.index(name),3:]);sr=Slerp(oldt,sr)(times) if len(raw)>1 else sr
        human_rot[name]=(sr*R.from_quat(zero[zn.index(name),3:]).inv()*rz).as_matrix()
    heading=np.arctan2(human_rot['Hips'][:,1,0],human_rot['Hips'][:,0,0])
    bp=[];br=[];com=[];points=[]
    foot_bodies=[m.body(s+('_ankle_pitch_link' if robot=='h2' else '_ankle_roll_link')).id for s in ('left','right')]
    for qi in q:
        d.qpos[:]=qi;mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);bp.append(d.xpos.copy());br.append(d.xmat.reshape(-1,3,3).copy());com.append(d.subtree_com[1].copy())
        feet=[]
        for side,b in zip(('left','right'),foot_bodies):
            if robot=='h2':
                g=m.geom(side+'_foot_collision').id;signs=np.array([[x,y,-1] for x in (-1,1) for y in (-1,1)])
                pts=(signs*m.geom_size[g])@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
            else:
                ids=np.flatnonzero((m.geom_bodyid==b)&(m.geom_type==mujoco.mjtGeom.mjGEOM_SPHERE)&(m.geom_contype!=0));assert len(ids)==4
                pts=d.geom_xpos[ids].copy();pts[:,2]-=m.geom_size[ids,0]
            feet.append(pts)
        points.append(feet)
    bp,br,com,points=map(np.asarray,(bp,br,com,points));torso=m.body('torso_link').id
    roll=lateral(br[:,torso],heading);expected=lateral(human_rot['Chest'],heading);roll_error=roll-expected
    center=points.mean((1,2));left=np.stack([-np.sin(heading),np.cos(heading),np.zeros_like(heading)],axis=1)
    centers=points.mean(2);width=np.sum((centers[:,0]-centers[:,1])*left,axis=-1)
    normalized=np.sum((com-center)*left,axis=-1)/width
    arms={}
    for side in ('Left','Right'):
        hp=human_pos[:,[names.index(side+s) for s in ('Arm','ForeArm','Hand')]]
        rp=bp[:,[m.body(side.lower()+'_'+s+'_link').id for s in ('shoulder_roll','elbow','wrist_yaw')]]
        hv=np.diff(hp,axis=1);rv=np.diff(rp,axis=1);lengths=np.linalg.norm(rv,axis=-1)
        hv/=np.linalg.norm(hv,axis=-1)[...,None];rv/=lengths[...,None]
        hflex=angle(hv[:,0],hv[:,1]);rflex=angle(rv[:,0],rv[:,1]);straight=hflex<=15
        desired=np.sum(hv*lengths[...,None],axis=1);actual=rp[:,2]-rp[:,0]
        hand_source=R.from_quat(raw[:,names.index(side+'Hand'),3:]);hand_source=Slerp(oldt,hand_source)(times) if len(raw)>1 else hand_source
        hand_target=hand_source*R.from_quat(scaler['joint_offsets'][side+'Hand'][1])
        wrist_error=np.rad2deg((hand_target.inv()*R.from_matrix(br[:,m.body(side.lower()+'_wrist_yaw_link').id])).magnitude())
        arms[side]=dict(wrist_orientation_error_deg=stats(wrist_error),upper_direction_error_deg=stats(angle(hv[:,0],rv[:,0])),forearm_direction_error_deg=stats(angle(hv[:,1],rv[:,1])),
          elbow_flex_error_deg=stats(abs(rflex-hflex)),normalized_hand_error=stats(np.linalg.norm(actual-desired,axis=-1)/lengths.sum(1)),
          human_straight_frames=int(straight.sum()),human_flex_when_straight_deg=stats(hflex[straight]),elbow_flex_error_when_human_straight_deg=stats(abs(rflex-hflex)[straight]),robot_flex_when_human_straight_deg=stats(rflex[straight]))
    foot={}
    for side,b in zip(('Left','Right'),foot_bodies):
        e=yaw_error(br[:,b],human_rot[side+'Foot'])
        target=source['targets'][:,source['names'].tolist().index(side+'Foot')]
        target_p=np.array([np.interp(times,oldt,target[:,i]) for i in range(3)]).T
        target_r=R.from_quat(target[:,3:]);target_r=Slerp(oldt,target_r)(times) if len(raw)>1 else target_r
        rotation_error=np.rad2deg((target_r.inv()*R.from_matrix(br[:,b])).magnitude())
        foot[side]=dict(position_error_mm=stats(np.linalg.norm(bp[:,b]-target_p,axis=1)*1000),rotation_error_deg=stats(rotation_error),yaw_error_deg=stats(abs(e)),signed_yaw_bias_deg=float(e.mean()),yaw_relative_pelvis_mean_deg=float(yaw_error(br[:,b],br[:,1]).mean()))
    limits=m.jnt_range[1:];excess=np.maximum(np.maximum(limits[:,0]-q[:,7:],q[:,7:]-limits[:,1]),0)
    native_q=data[:,columns];native_excess=np.maximum(np.maximum(limits[:,0]-native_q,native_q-limits[:,1]),0)
    native_step=abs(np.diff(native_q,axis=0))
    arm_indices=[i for i,j in enumerate(range(1,m.njnt)) if any(x in m.joint(j).name for x in ('shoulder','elbow','wrist'))]
    native_checks=dict(fps=native,joint_limit_excess_deg=float(np.rad2deg(native_excess.max())),
        max_joint_step_deg=float(np.rad2deg(native_step.max())) if len(native_step) else 0.,
        max_joint_speed_rad_s=float(native_step.max()*native) if len(native_step) else 0.,
        max_arm_speed_rad_s=float(native_step[:,arm_indices].max()*native) if len(native_step) else 0.)
    if robot=='h2':
        urdf=ROOT/'gear_sonic/data/assets/robot_description/urdf/h2/h2.urdf'
        speeds={j.get('name'):float(j.find('limit').get('velocity')) for j in ET.parse(urdf).getroot().findall('joint') if j.find('limit') is not None}
        caps=np.array([speeds[m.joint(i).name] for i in range(1,m.njnt)])
        ratios=native_step*native/caps if len(native_step) else np.zeros((1,len(caps)))
        native_checks['max_urdf_velocity_ratio']=float(ratios.max())
        native_checks['max_arm_urdf_velocity_ratio']=float(ratios[:,arm_indices].max())
        margin_by_joint={}
        for j in arm_indices:
            name=m.joint(1+j).name
            if not any(k in name for k in ('shoulder','wrist')):continue
            margin=np.rad2deg(np.minimum(native_q[:,j]-limits[j,0],limits[j,1]-native_q[:,j]))
            margin_by_joint[name]=dict(min_margin_deg=float(margin.min()),frames_below_2deg=int((margin<2-1e-6).sum()))
        native_checks['shoulder_wrist_min_limit_margin_deg']=min(v['min_margin_deg'] for v in margin_by_joint.values())
        native_checks['shoulder_wrist_joint_margins']=margin_by_joint
    z=points[...,2].min((1,2));z=z-z[0]
    return dict(robot=robot,source=str(path.resolve()),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),fps=fps,frames=len(times),native_frames=len(data),native_checks=native_checks,
        torso_orientation_error_deg=stats(np.rad2deg((R.from_matrix(human_rot['Chest']).inv()*R.from_matrix(br[:,torso])).magnitude())),torso_lateral_mean_deg=float(roll.mean()),torso_lateral_error_deg=stats(abs(roll_error)),torso_signed_error_deg=float(roll_error.mean()),
        com_lateral_over_stance_width_mean=float(normalized.mean()),arms=arms,feet=foot,
        joint_limit_excess_deg=float(np.rad2deg(excess.max())),max_joint_step_deg=float(np.rad2deg(abs(np.diff(q[:,7:],axis=0)).max())) if len(q)>1 else 0.,
        sole_min_relative_first_frame_mm=float(z.min()*1000),scope=f'FK at {fps:g}Hz, joint limits and speeds also checked at native rate; no measured load or policy tracking.')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--csv-dir',type=Path,required=True);p.add_argument('--trace-dir',type=Path,required=True)
    p.add_argument('--robot',choices=['h2','g1'],required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--fps',type=float,default=50.);a=p.parse_args()
    result=[]
    for path in sorted(a.csv_dir.glob('*.csv')):
        trace=a.trace_dir/(path.stem+'_trace.npz')
        if not trace.exists():raise ValueError('Missing source trace '+str(trace))
        r=evaluate(path,trace,a.robot,a.reference,a.fps);r['key']=path.stem;result.append(r)
        print(json.dumps(dict(key=path.stem,roll=r['torso_lateral_mean_deg'],roll_error=r['torso_lateral_error_deg']['mean'],arm_flex_error={s:v['elbow_flex_error_deg']['mean'] for s,v in r['arms'].items()},hand_error={s:v['normalized_hand_error']['mean'] for s,v in r['arms'].items()},foot_yaw_error={s:v['yaw_error_deg']['mean'] for s,v in r['feet'].items()})),flush=True)
    a.out.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
