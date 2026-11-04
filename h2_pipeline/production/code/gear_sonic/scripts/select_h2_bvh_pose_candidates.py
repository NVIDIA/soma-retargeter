"""Select complete kinematic IK branches, preserving frame IDs and source goals.

The G1 reference defines admission errors; URDF limits independently exclude
invalid motion. Arm branches may be combined only for identical torso poses.
"""
import argparse,csv,json,hashlib,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from gear_sonic.scripts.audit_g1_style_retarget import evaluate

ARM_METRICS=('upper_direction_error_deg','forearm_direction_error_deg','elbow_flex_error_deg','normalized_hand_error','wrist_orientation_error_deg','elbow_flex_error_when_human_straight_deg')
FOOT_METRICS=('yaw_error_deg','rotation_error_deg','position_error_mm')


def read(path):
    with path.open() as f:r=csv.reader(f);header=next(r);raw=np.array(list(r),float)
    return header,raw


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True);p.add_argument('--foot-candidates',type=Path,nargs='+',default=[]);p.add_argument('--arm-candidates',type=Path,nargs='+',required=True);p.add_argument('--upper-body-candidates',type=Path,nargs='+',default=[])
    p.add_argument('--g1-audits',type=Path,nargs='+',required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--joint-limit-margin-deg',type=float,default=2.);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);(a.out/'csv').mkdir();records=[]
    g1={x['key']:x for f in a.g1_audits for x in json.loads(f.read_text())}
    urdf=Path(__file__).resolve().parents[2]/'gear_sonic/data/assets/robot_description/urdf/h2/h2.urdf'
    velocity={j.get('name'):float(j.find('limit').get('velocity')) for j in ET.parse(urdf).getroot().findall('joint') if j.find('limit') is not None}
    audits={}
    def audit(path,trace):
        if str(path) not in audits:audits[str(path)]=evaluate(path,trace,'h2',a.reference)
        return audits[str(path)]
    for base in sorted((a.base/'csv').glob('*.csv')):
        key=base.stem;header,original=read(base);clean=[x.removesuffix('_dof') for x in header]
        trace=a.base/(key+'_trace.npz');fps=float(np.load(trace)['fps']);lower=[i for i,n in enumerate(clean) if any(k in n for k in ('hip','knee','ankle','waist'))]
        def speed_ok(raw,cols):
            caps=np.array([velocity[clean[i]] for i in cols]);return len(raw)<2 or float(np.max(abs(np.diff(raw[:,cols],axis=0))*fps/caps))<=1+1e-6
        def tracking(values,baseline,metrics):
            active=[k for k in metrics if values[k] is not None]
            if any(baseline[k] is None for k in active):raise ValueError('Source straight mask mismatch')
            checks=[(values[k][z],baseline[k][z]) for k in active for z in ('mean','p95')]
            passed=all(x<=y+1e-6 for x,y in checks)
            worst=max(x/(y+1e-6) for x,y in checks)
            max_ratio=max(values[k]['max']/(baseline[k]['max']+1e-6) for k in active)
            return passed,worst,max_ratio
        feet=[]
        for folder in [a.base,*a.foot_candidates]:
            path=folder/'csv'/base.name
            if not path.exists():continue
            h,raw=read(path)
            if h!=header or raw.shape!=original.shape or not np.array_equal(raw[:,0],original[:,0]):raise ValueError('Frame/header mismatch')
            r=audit(path,trace);checks=[tracking(r['feet'][side],g1[key]['feet'][side],FOOT_METRICS) for side in ('Left','Right')]
            valid=speed_ok(raw,lower) and r['joint_limit_excess_deg']<=1e-4
            feet.append((not(valid and all(x[0] for x in checks)),max(x[2] for x in checks),max(x[1] for x in checks),path,raw))
        feet.sort(key=lambda x:x[:3]);bad,_,_,foot_path,result=feet[0];result=result.copy()
        if bad:raise ValueError('No admitted foot branch: '+key)
        chosen={};tested={};chosen_peaks=[];waist_source=None
        torso_cols=[4,5,6]+[i for i,n in enumerate(clean) if 'waist' in n]
        for side in ('Left','Right'):
            cols=[i for i,n in enumerate(clean) if n.startswith(side.lower()+'_') and any(k in n for k in ('shoulder','elbow','wrist'))];candidates=[];tested[side]=[]
            for folder in a.arm_candidates:
                path=folder/'csv'/base.name
                if not path.exists():continue
                h,raw=read(path)
                if h!=header or raw.shape!=original.shape or not np.array_equal(raw[:,0],original[:,0]):raise ValueError('Frame/header mismatch')
                if not np.allclose(raw[:,torso_cols],result[:,torso_cols],atol=1e-9,rtol=0):continue
                r=audit(path,trace);ok,score,peak=tracking(r['arms'][side],g1[key]['arms'][side],ARM_METRICS)
                margin=min(v['min_margin_deg'] for n,v in r['native_checks']['shoulder_wrist_joint_margins'].items() if n.startswith(side.lower()+'_'))
                valid=ok and speed_ok(raw,cols) and r['joint_limit_excess_deg']<=1e-4 and margin>=a.joint_limit_margin_deg-1e-6
                tested[side].append(dict(path=str(path),admitted=valid,minimum_joint_margin_deg=margin,mean_p95_max_ratio=score,worst_frame_max_ratio=peak))
                candidates.append((not valid,peak,score,path,raw))
            candidates.sort(key=lambda x:x[:3])
            if not candidates or candidates[0][0]:raise ValueError('No admitted arm branch: '+key+' '+side+' '+json.dumps(tested[side]))
            _,peak,_,path,raw=candidates[0];result[:,cols]=raw[:,cols];chosen[side]=str(path);chosen_peaks.append(peak)
        upper_cols=[i for i,n in enumerate(clean) if any(k in n for k in ('waist','shoulder','elbow','wrist'))]
        fixed=[i for i in range(len(header)) if i not in upper_cols];best_peak=max(chosen_peaks);upper_tested=[]
        for folder in a.upper_body_candidates:
            path=folder/'csv'/base.name
            if not path.exists():continue
            h,raw=read(path)
            if h!=header or raw.shape!=result.shape or not np.allclose(raw[:,fixed],result[:,fixed],atol=1e-9,rtol=0):continue
            r=audit(path,trace);checks=[tracking(r['arms'][side],g1[key]['arms'][side],ARM_METRICS) for side in ('Left','Right')]
            checks.append(tracking(r,g1[key],('torso_lateral_error_deg','torso_orientation_error_deg')))
            valid=r['native_checks']['shoulder_wrist_min_limit_margin_deg']>=a.joint_limit_margin_deg-1e-6 and all(x[0] for x in checks) and speed_ok(raw,upper_cols) and r['native_checks']['joint_limit_excess_deg']<=1e-4
            peak=max(x[2] for x in checks);upper_tested.append(dict(path=str(path),admitted=valid,worst_frame_max_ratio=peak))
            if valid and peak<best_peak:
                result[:,upper_cols]=raw[:,upper_cols];chosen={side:str(path) for side in ('Left','Right')};waist_source=str(path);best_peak=peak
        target=a.out/'csv'/base.name
        with target.open('w') as f:w=csv.writer(f);w.writerow(header);w.writerows(result)
        (a.out/trace.name).symlink_to(trace.resolve())
        provenance={str(path):hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in [str(foot_path),*chosen.values()]}
        row=dict(key=key,feet=str(foot_path),arms=chosen,waist_source=waist_source,upper_body_candidate_checks=upper_tested,candidate_checks=tested,input_sha256=provenance,output_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),frame_ids_preserved=True,torso_rotation_compatibility_checked=True)
        records.append(row);print(json.dumps(dict(key=key,feet=str(foot_path),arms=chosen)),flush=True)
    (a.out/'selection.json').write_text(json.dumps(dict(samples=records,required_shoulder_wrist_margin_deg=a.joint_limit_margin_deg,scope=__doc__,published=False,training_started=False),indent=2)+'\n')
    (a.out/'runner.py').write_bytes(Path(__file__).read_bytes())
if __name__=='__main__':main()
