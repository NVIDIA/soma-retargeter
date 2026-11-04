"""Independent frozen MuJoCo audit of experimental root-changing candidates.

Always reports root-contract change separately. Does not promote any motion.
"""
from pathlib import Path
import argparse,importlib.util,json,sys
import numpy as np
from scipy.spatial.transform import Rotation as R,Slerp
from native_projection import BASE,KEY,FootModel,load,sample,compare_geometry,digest,save

def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

def main():
    p=argparse.ArgumentParser();p.add_argument('candidates',type=Path,nargs='+');a=p.parse_args()
    here=BASE/'workspace/GR00T-WholeBodyControl-h2-clean/gear_sonic/scripts';c=module('v10_independent_common',here/'run_h2_foot_preserving_reopt.py')
    bundle=BASE/'artifacts/h2_bvh_gpu_staged_v5';source=BASE/'artifacts/h2_bvh_gpu_refine131k_116014_v6';api=c.api_for(bundle);signature=api.verify_bundle();job=next(j for j in api.jobs_for(source) if j['key']==KEY)
    from gear_sonic.scripts.audit_g1_style_retarget import evaluate
    from gear_sonic.scripts.h2_bvh_reference_worker import compare,ARM_METRICS,FOOT_METRICS
    assert api.raw_valid(Path(job['work']),job)
    baseline=source/'refined/csv'/f'{KEY}.csv';old=c.read(source/'reports'/f'{KEY}.json');assert digest(baseline)==old['refined_sha256']
    fm=FootModel(bundle/'code/gear_sonic/data/assets/robot_description/mjcf/h2.xml');_,raw,_,original=load(baseline,fm)
    trace=np.load(Path(job['work'])/'h2'/f'{KEY}_trace.npz');ref=np.load(bundle/'profile/reference.npz');zero=ref['source'];names=ref['source_names'].tolist();sn=trace['source_names'].tolist()
    heading=R.from_euler('z',(R.from_quat(zero[names.index('Hips'),3:])*R.from_quat([.5,.5,.5,.5])).as_euler('xyz')[2])
    human=R.from_quat(trace['source_world'][:,sn.index('Hips'),3:])*R.from_quat(zero[names.index('Hips'),3:]).inv()*heading
    import csv
    with (Path(job['work'])/'g1/csv'/f'{KEY}.csv').open() as f:
        rd=csv.reader(f);next(rd);graw=np.array(list(rd),float)
    grot=R.from_euler('xyz',graw[:,4:7]);native_times=np.arange(len(raw))/120.
    catalogs=[BASE/'artifacts'/n/'catalog.json' for n in ['h2_accepted52217_recovery_v6','h2_accepted52219_compound_v9']];pins={str(p):digest(p) for p in catalogs}
    for candidate in a.candidates:
        before=digest(candidate);_,data,_,q=load(candidate,fm);assert q.shape==original.shape
        audits={};checks=[];root={};physical=True
        for rate in [120,50]:
            rr=str(rate);h=evaluate(candidate,Path(job['work'])/'h2'/f'{KEY}_trace.npz','h2',Path(job['reference']),rate);audits[rr]=h;g=old['audit_g1'][rr]
            for side in ['Left','Right']:
                for section,metrics in [('arms',ARM_METRICS),('feet',FOOT_METRICS)]:
                    checks.extend(dict(fps=rate,side=side,section=section,metric=k,stat=stat,h2=x,g1=y,passed=ok) for k,stat,x,y,ok in compare(h[section][side],g[section][side],metrics))
            checks.extend(dict(fps=rate,section='torso',metric=k,stat=stat,h2=x,g1=y,passed=ok) for k,stat,x,y,ok in compare(h,g,('torso_lateral_error_deg','torso_orientation_error_deg')))
            nc=h['native_checks'];physical &= nc['max_urdf_velocity_ratio']<=1+1e-6 and nc['joint_limit_excess_deg']<=1e-4 and nc['shoulder_wrist_min_limit_margin_deg']>=2-1e-6
            times=np.arange(int(np.floor(native_times[-1]*rate+1e-8))+1)/rate;hr=Slerp(native_times,human)(times);gr=Slerp(native_times,grot)(times);cr=R.from_quat(sample(q,times)[:,[4,5,6,3]])
            ce=(hr.inv()*cr).magnitude();ge=(hr.inv()*gr).magnitude();cs=(cr[:-1].inv()*cr[1:]).magnitude()*rate;gs=(gr[:-1].inv()*gr[1:]).magnitude()*rate
            root[rr]=dict(candidate_error_deg={name:float(fun(ce)*180/np.pi) for name,fun in [('mean',np.mean),('p95',lambda v:np.percentile(v,95)),('max',np.max)]},g1_error_deg={name:float(fun(ge)*180/np.pi) for name,fun in [('mean',np.mean),('p95',lambda v:np.percentile(v,95)),('max',np.max)]},candidate_angular_speed=float(cs.max()),g1_angular_speed=float(gs.max()),passed=bool(all(fun(ce)<=fun(ge)+1e-8 for fun in [np.mean,lambda v:np.percentile(v,95),np.max]) and cs.max()<=gs.max()+1e-6))
        geometry=compare_geometry(fm,original,q);failed=[v for v in checks if not v['passed']]
        foot_legacy=all(v['legacy']['position_mm']<=.5 and v['legacy']['rotation_deg']<=.1 and v['legacy_zero_clearance_crossings']==0 for v in geometry.values())
        foot_native=all(v['native_pose']['position_mm']<=.5 and v['native_pose']['rotation_deg']<=.1 for v in geometry.values())
        unchanged_root=np.array_equal(data[:,4:7],raw[:,4:7]);root_ok=all(v['passed'] for v in root.values())
        report=dict(key=KEY,candidate=str(candidate),candidate_sha256=before,baseline_sha256=digest(baseline),bundle_sha256=signature,code_sha256=digest(__file__),audit_h2=audits,failed_g1_checks=failed,g1_pass=not failed,physical_pass=bool(physical),root=root,root_metrics_pass=root_ok,geometry=geometry,legacy_foot_guard_pass=foot_legacy,native_foot_geometry_pass=foot_native,root_rotation_contract_exact=bool(unchanged_root),legacy_admission_pass=bool(physical and not failed and foot_legacy and unchanged_root),root_changed_kinematic_all_pass=bool(physical and not failed and foot_legacy and root_ok),admission_performed=False,training_started=False)
        assert before==digest(candidate) and all(digest(p)==v for p,v in pins.items())
        save(candidate.parent/'independent_audit.json',report)
        print(json.dumps({k:v for k,v in report.items() if k not in ['audit_h2','failed_g1_checks','root']}|{'g1_failed_check_count':len(failed)}),flush=True)

if __name__=='__main__':main()
