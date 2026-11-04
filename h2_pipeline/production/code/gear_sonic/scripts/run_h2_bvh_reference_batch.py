"""Two-stage full BVH retarget: batched Newton GPU IK, then CPU reference refinement.

H2 only: existing G1 CSVs supply the comparison motion; G1 IK is never run.
512 = at most 256 simultaneous Newton motions on each of two GPUs. The MuJoCo/SciPy
refinement is CPU work. Accepted CSV and needs-review CSV are kept separate.
"""
import argparse,concurrent.futures,fcntl,hashlib,json,os,shutil,signal,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
BUNDLE=ROOT.parent


def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def atomic_json(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)

def verify_bundle():
    manifest=json.loads((BUNDLE/'bundle_manifest.json').read_text())
    for rel,sha in manifest['files'].items():
        if digest(BUNDLE/rel)!=sha:raise ValueError('Bundle changed: '+rel)
    return digest(BUNDLE/'bundle_manifest.json')

def prepare(a):
    signature=verify_bundle();g1_paths={}
    if not a.g1_csv_root:raise ValueError('--g1-csv-root is required; G1 retarget fallback is disabled')
    for path in a.g1_csv_root.rglob('*.csv'):
        if path.stem in g1_paths:raise ValueError('Duplicate G1 CSV basename: '+path.stem)
        g1_paths[path.stem]=path
    selection=json.loads(a.source_selection.read_text())['motions'];paths={}
    for path in a.bvh_root.rglob('*.bvh'):
        if path.stem in paths:raise ValueError('Duplicate BVH basename: '+path.stem)
        paths[path.stem]=path
    if a.keys:selection=[m for m in selection if m['key'] in a.keys]
    if a.limit:selection=selection[:a.limit]
    if not selection:raise ValueError('Empty selection')
    missing=[m['key'] for m in selection if m['key'] not in paths]
    if missing:raise ValueError('Missing original BVH: '+str(missing[:20]))
    motions=[]
    for m in selection:
        path=paths[m['key']].resolve();st=path.stat();g1=g1_paths.get(m['key'])
        if g1 is None:raise ValueError('Missing existing G1 CSV: '+m['key'])
        gs=g1.stat();motions.append(dict(key=m['key'],bvh=str(path),size=st.st_size,mtime_ns=st.st_mtime_ns,g1_csv=str(g1.resolve()),g1_size=gs.st_size,g1_mtime_ns=gs.st_mtime_ns))
    a.out.mkdir(parents=True,exist_ok=False)
    atomic_json(a.out/'manifest.json',dict(schema='h2_bvh_reference_existing_g1.v2',g1_retarget=False,g1_csv_units='cm_degrees',bundle=str(BUNDLE),bundle_sha256=signature,source_selection=str(a.source_selection),source_selection_sha256=digest(a.source_selection),motions=motions,scope=__doc__,training_started=False))
    print(json.dumps(dict(prepared=len(motions),out=str(a.out))),flush=True)

def post(a):
    from gear_sonic.scripts.h2_bvh_reference_worker import process_one
    jobs=json.loads(a.jobs.read_text());counts=dict(accepted=0,needs_review=0,error=0)
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.cpu_workers) as pool:
        futures=[pool.submit(process_one,job) for job in jobs]
        for f in concurrent.futures.as_completed(futures):
            result=f.result();counts[result['status']]+=1
            atomic_json(a.jobs.parent/'post_progress.json',dict(completed=sum(counts.values()),counts=counts,total=len(jobs)))
            print(json.dumps(result),flush=True)
    if counts['error']:print(json.dumps(dict(post_errors=counts['error'])),flush=True)

SOURCE_KEEP = ('Hips','Chest','LeftFoot','RightFoot','LeftArm','LeftForeArm','LeftHand','RightArm','RightForeArm','RightHand')


def compact_trace(source_world, source_names):
    """Lossless subset: all source joints consumed by IK and the existing audit."""
    names=list(source_names)
    return source_world[:,[names.index(n) for n in SOURCE_KEEP]],list(SOURCE_KEEP)


def raw_valid(work, job):
    for rel,sha in job.get('raw_files',{}).items():
        path=work/rel
        if not path.is_file() or digest(path)!=sha:return False
    return len(job.get('raw_files',{}))==4


def seal_jobs(work,jobs):
    for job in jobs:
        key=job['key'];job['raw_files']={str(rel):digest(work/rel) for robot in ('h2','g1') for rel in (Path(robot)/'csv'/f'{key}.csv',Path(robot)/f'{key}_trace.npz')}
    atomic_json(work/'jobs.json',jobs)


def gpu(a):
    sys.path.insert(0,str(BUNDLE/'soma-retargeter-h2'))
    import numpy as np,warp as wp
    from soma_retargeter.assets import bvh,csv
    from soma_retargeter.pipelines.newton_pipeline import NewtonPipeline
    from soma_retargeter.pipelines import utils
    from soma_retargeter.robotics.human_to_robot_scaler import HumanToRobotScaler
    from gear_sonic.scripts.h2_existing_g1_reference import normalize_existing_g1
    from soma_retargeter.utils import newton_utils
    from soma_retargeter.utils.space_conversion_utils import SpaceConverter,get_facing_direction_type_from_str
    from gear_sonic.scripts.h2_kinematic_targets import segment_targets
    manifest=json.loads((a.out/'manifest.json').read_text());items=manifest['motions'][a.shard::a.shards];counts=dict(accepted=0,needs_review=0,error=0);done=0;reused=0;approved=json.loads((BUNDLE/'approved/manifest.json').read_text());start=time.time()
    total_items=len(items)
    if manifest.get('length_plan_sha256'):
        if not a.raw_only or a.shards!=2 or a.batch_size!=256:raise ValueError('Length plan requires raw-only, two GPUs and 256 motions each')
        plan_path=a.out/'length_plan.json'
        if digest(plan_path)!=manifest['length_plan_sha256']:raise ValueError('Length plan changed')
        plan=json.loads(plan_path.read_text());by_key={m['key']:m for m in items}
        pending=plan['shards'][a.shard];imported=set(plan['imported_keys'])
        if set(pending)!=(set(by_key)-imported) or len(pending)!=len(set(pending)):raise ValueError('Length plan coverage mismatch')
        items=[by_key[k] for k in pending];done=total_items-len(items);reused=done

    state=a.out/f'shard_{a.shard}.json'
    def update(phase,**extra):atomic_json(state,dict(shard=a.shard,gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),phase=phase,completed=done,reused=reused,total=total_items,counts=counts,seconds=round(time.time()-start,1),**extra))
    wp.init();update('initializing')
    with wp.ScopedDevice('cuda:0'):
        for batch_no,base in enumerate(range(0,len(items),a.batch_size)):
            work=a.out/'work'/f'gpu{a.shard}_batch{batch_no:06d}';pending=[];cached=[]
            raw_jobs={j['key']:j for j in json.loads((work/'jobs.json').read_text())} if a.raw_only and (work/'jobs.json').exists() else {}
            for item in items[base:base+a.batch_size]:
                key=item['key'];src=Path(item['bvh']);st=src.stat()
                if st.st_size!=item['size'] or st.st_mtime_ns!=item['mtime_ns']:raise ValueError('BVH changed since prepare: '+key)
                if a.raw_only and key in raw_jobs and raw_valid(work,raw_jobs[key]):
                    job=raw_jobs[key];g1=Path(item['g1_csv']);gs=g1.stat()
                    if (gs.st_size,gs.st_mtime_ns)!=(item['g1_size'],item['g1_mtime_ns']):raise ValueError('Existing G1 CSV changed: '+key)
                    if job['bvh_sha256']!=digest(src):raise ValueError('Cached BVH mismatch: '+key)
                    cached.append(job);continue
                report_path=a.out/'reports'/f'{key}.json'
                if not a.raw_only and report_path.exists():
                    report=json.loads(report_path.read_text())
                    if report['status'] in ('accepted','needs_review') and Path(report['output']).exists() and digest(report['output'])==report['output_sha256']:
                        counts[report['status']]+=1;done+=1;reused+=1;continue
                sha=digest(src);item=dict(item,bvh_sha256=sha)
                frozen=approved.get(key)
                if not a.raw_only and frozen and sha==frozen['bvh_sha256']:
                    dst=a.out/'accepted/csv'/f'{key}.csv';dst.parent.mkdir(parents=True,exist_ok=True);src_csv=BUNDLE/'approved/csv'/f'{key}.csv'
                    if digest(src_csv)!=frozen['csv_sha256']:raise ValueError('Approved reference changed: '+key)
                    shutil.copyfile(src_csv,dst.with_suffix('.partial'));dst.with_suffix('.partial').replace(dst)
                    atomic_json(report_path,dict(key=key,status='accepted',source_bvh_sha256=sha,output=str(dst),output_sha256=digest(dst),method='reuse_user_reviewed_reference_for_identical_BVH_and_model',approved_exception=frozen.get('approved_exception'),training_started=False));done+=1;counts['accepted']+=1;reused+=1;continue
                pending.append(item)
            if not pending:
                if a.raw_only:done+=len(cached);reused+=len(cached)
                update('raw_batch_complete' if a.raw_only else 'batch_complete',batch=batch_no);continue
            if shutil.disk_usage(a.out).free<20*1024**3:raise RuntimeError('Less than 20 GiB free; stopped before next batch')
            work.mkdir(parents=True,exist_ok=True);update('loading_bvh',batch=batch_no,work=str(work),batch_motions=len(pending))
            groups=[]
            for item in pending:
                sk,anim=bvh.load_bvh(item['bvh'])
                for group in groups:
                    ref=group[0]
                    if list(sk.joint_names)==list(ref.joint_names) and np.array_equal(sk.parent_indices,ref.parent_indices) and np.allclose(sk._reference_local_transforms,ref._reference_local_transforms,rtol=0,atol=1e-6):
                        anim.skeleton=ref  # Topology and bind pose were verified above.
                        group[1].append(item);group[2].append(anim);break
                else:groups.append((sk,[item],[anim]))
            print(json.dumps(dict(batch=batch_no,skeleton_groups=[len(g[1]) for g in groups])),flush=True)
            for skeleton,group_items,animations in groups:
                conv=SpaceConverter(get_facing_direction_type_from_str('Mujoco')).transform(wp.transform_identity())
                source_data=None
                for robot,target in [('h2','h2')]:
                    update('gpu_'+robot,batch=batch_no,work=str(work),batch_motions=len(group_items))
                    model=ROOT/'gear_sonic/data/assets/robot_description/mjcf'/('h2.xml' if robot=='h2' else 'g1_29dof_rev_1_0.xml');utils.get_robot_mjcf_path=lambda target:model
                    config=BUNDLE/'profile/retarget.json' if robot=='h2' else BUNDLE/'soma-retargeter-h2/soma_retargeter/configs/unitree_g1/soma_to_g1_retargeter_config.json'
                    cfg=json.loads(config.read_text());pipe=NewtonPipeline(skeleton,'soma',target,retarget_config=cfg);pipe.gpu_transfer_batching=bool(manifest.get('gpu_transfer_batching',False));pipe.add_input_motions(animations,[conv]*len(animations),True);trim=pipe.num_initialization_frames+pipe.num_stabilization_frames
                    if source_data is None:
                        source_data=[]
                        for anim in animations:
                            buff=newton_utils.create_buffer_with_initialization_frames(pipe.initialization_pose,anim,pipe.num_initialization_frames,pipe.num_stabilization_frames)
                            source_data.append(np.asarray([buff.compute_global_transforms(i,root_tx=conv) for i in range(buff.num_frames)],float))
                    if cfg.get('h2_segment_targets'):
                        for j,raw in enumerate(source_data):pipe.input_targets[j]=segment_targets(pipe.input_targets[j],pipe.mapped_joints,raw,list(skeleton.joint_names),cfg['h2_segment_targets'])
                    results=pipe.execute();dest=work/robot;(dest/'csv').mkdir(parents=True,exist_ok=True)
                    if len(results)!=len(group_items):raise ValueError('GPU batch output count mismatch')
                    for j,(item,anim,result) in enumerate(zip(group_items,animations,results)):
                        if result.num_frames!=anim.num_frames:raise ValueError('GPU frame count mismatch')
                        key=item['key'];csv.save_csv(str(dest/'csv'/f'{key}.csv'),result,csv_config=csv.get_csv_config(target))
                        trace_source,trace_names=compact_trace(source_data[j][trim:],skeleton.joint_names) if a.raw_only else (source_data[j][trim:],skeleton.joint_names)
                        np.savez_compressed(dest/f'{key}_trace.npz',source_world=trace_source,source_names=np.array(trace_names),targets=pipe.input_targets[j][trim:],names=np.array(pipe.mapped_joints),fps=anim.sample_rate)
                    del pipe,results
                # Goal transforms only. No G1 NewtonPipeline/model/IK is instantiated.
                update('load_existing_g1',batch=batch_no,work=str(work),batch_motions=len(group_items))
                g1_scaler=HumanToRobotScaler(skeleton,1.70,str(BUNDLE/'soma-retargeter-h2/soma_retargeter/configs/unitree_g1/soma_to_g1_scaler_config.json'))
                dest=work/'g1';(dest/'csv').mkdir(parents=True,exist_ok=True)
                for j,(item,anim) in enumerate(zip(group_items,animations)):
                    source=Path(item['g1_csv']);st=source.stat()
                    if (st.st_size,st.st_mtime_ns)!=(item['g1_size'],item['g1_mtime_ns']):raise ValueError('Existing G1 CSV changed: '+item['key'])
                    normalize_existing_g1(source,dest/'csv'/f"{item['key']}.csv",anim.num_frames)
                    goals=g1_scaler.compute_effectors_from_buffer(anim,True,conv)
                    trace_source,trace_names=compact_trace(source_data[j][trim:],skeleton.joint_names) if a.raw_only else (source_data[j][trim:],skeleton.joint_names)
                    np.savez_compressed(dest/f"{item['key']}_trace.npz",source_world=trace_source,source_names=np.array(trace_names),targets=goals,names=np.array(g1_scaler.effector_names()),fps=anim.sample_rate)
                    item['g1_csv_sha256']=digest(source)
            jobs=[dict(item,work=str(work),out=str(a.out),reference=str(BUNDLE/'profile/reference.npz')) for item in pending]
            if a.raw_only:
                seal_jobs(work,cached+jobs);done+=len(cached)+len(jobs);reused+=len(cached)
                update('raw_batch_complete',batch=batch_no,work=str(work),batch_motions=len(cached)+len(jobs));continue
            atomic_json(work/'jobs.json',jobs);update('cpu_refinement',batch=batch_no,work=str(work),batch_motions=len(pending))
            command=[a.cpu_python,'-u','-m','gear_sonic.scripts.run_h2_bvh_reference_batch','post','--jobs',str(work/'jobs.json'),'--cpu-workers',str(a.cpu_workers)]
            with (a.out/'logs'/f'cpu_{a.shard}_{batch_no:06d}.log').open('w') as log:subprocess.run(command,cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,check=True)
            for item in pending:
                report=json.loads((a.out/'reports'/f"{item['key']}.json").read_text());counts[report['status']]+=1;done+=1
            update('batch_complete',batch=batch_no)
            # Only this run's disposable raw batch is removed; final reports/CSVs persist.
            shutil.rmtree(work)
        update('complete')

def launch(a):
    def interrupted(signum,frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,interrupted)
    signature=verify_bundle();manifest=json.loads((a.out/'manifest.json').read_text())
    if signature!=manifest['bundle_sha256']:raise ValueError('Run belongs to another bundle')
    gpus=a.gpus.split(',')
    if not gpus or any(not x.isdecimal() for x in gpus) or len(set(gpus))!=len(gpus) or a.batch_size_total%len(gpus) or a.cpu_workers<len(gpus):raise ValueError('Invalid GPU/batch/CPU partition')
    lock=(a.out/'launch.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (a.out/'logs').mkdir(exist_ok=True);children=[];started=time.time()
    try:
        for shard,gpu_id in enumerate(gpus):
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu_id,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1',PYTHONPATH=str(ROOT)+os.pathsep+str(BUNDLE/'soma-retargeter-h2'),SOMA_RETARGETER_ROOT=str(BUNDLE/'soma-retargeter-h2'))
            cmd=[a.gpu_python,'-u','-m','gear_sonic.scripts.run_h2_bvh_reference_batch','gpu','--out',str(a.out),'--shard',str(shard),'--shards',str(len(gpus)),'--batch-size',str(a.batch_size_total//len(gpus)),'--cpu-workers',str(max(1,a.cpu_workers//len(gpus))),'--cpu-python',a.cpu_python]
            if a.raw_only:cmd.append('--raw-only')
            log=(a.out/'logs'/f'gpu_{gpu_id}.log').open('a');children.append((subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True),log))
        while True:
            states=[]
            for shard in range(len(gpus)):
                path=a.out/f'shard_{shard}.json'
                if path.exists():
                    state=json.loads(path.read_text());post_path=Path(state.get('work','/nonexistent'))/'post_progress.json'
                    if state['phase']=='cpu_refinement' and post_path.exists():
                        post_state=json.loads(post_path.read_text());state['completed']+=post_state['completed'];state['counts']={k:state['counts'][k]+post_state['counts'][k] for k in state['counts']}
                    states.append(state)
            progress=dict(stage='gpu_retarget' if a.raw_only else 'combined',g1_retarget=False,g1_reference='existing_csv',total=len(manifest['motions']),completed=sum(s['completed'] for s in states),counts={k:sum(s['counts'][k] for s in states) for k in ['accepted','needs_review','error']},gpus=gpus,gpu_batch_size_total=a.batch_size_total,cpu_workers=a.cpu_workers,seconds=round(time.time()-started,1),shards=states,training_started=False)
            atomic_json(a.out/'progress.json',progress);print(json.dumps(progress),flush=True)
            failed=[p.returncode for p,l in children if p.poll() is not None and p.returncode!=0]
            if failed:raise RuntimeError('Worker failed; inspect logs. Exit codes: '+str(failed))
            if all(p.poll() is not None for p,l in children):break
            time.sleep(5)
        if progress['completed']!=progress['total']:raise RuntimeError('Incomplete run')
        atomic_json(a.out/('raw_complete.json' if a.raw_only else 'complete.json'),progress)
    finally:
        for proc,log in children:
            if proc.poll() is None:
                try:os.killpg(proc.pid,signal.SIGTERM)
                except ProcessLookupError:pass
        for proc,log in children:proc.wait();log.close()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','launch','gpu','post','check']);p.add_argument('--out',type=Path);p.add_argument('--bvh-root',type=Path);p.add_argument('--source-selection',type=Path);p.add_argument('--g1-csv-root',type=Path);p.add_argument('--keys',nargs='+');p.add_argument('--limit',type=int);p.add_argument('--gpus',default='0,1');p.add_argument('--batch-size-total',type=int,default=512);p.add_argument('--batch-size',type=int,default=256);p.add_argument('--cpu-workers',type=int,default=32);p.add_argument('--gpu-python',default='/home/arrc/.conda/envs/soma-retargeter/bin/python');p.add_argument('--cpu-python',default='/home/arrc/.conda/envs/groot-wbc/bin/python');p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=2);p.add_argument('--jobs',type=Path);p.add_argument('--raw-only',action='store_true');a=p.parse_args()
    if a.out:a.out=a.out.resolve()
    if a.mode=='check':print(json.dumps(dict(bundle_sha256=verify_bundle(),scope=__doc__)));return
    if a.mode in ('prepare','launch','gpu') and not a.out:p.error('--out required')
    if a.mode=='prepare' and (not a.bvh_root or not a.source_selection):p.error('prepare requires --bvh-root and --source-selection')
    if a.mode=='post' and not a.jobs:p.error('post requires --jobs')
    globals()[a.mode](a)
if __name__=='__main__':main()
