"""Separated dataset stages: all GPU retarget, all GPU refinement, then FK audit.

GPU results are candidates. Admission uses the unchanged independent G1/H2 audit.
No G1 retarget, CPU optimizer fallback, training, or hardware execution.
"""
import argparse,concurrent.futures,fcntl,json,os,shutil,signal,subprocess,sys,time
from pathlib import Path
from gear_sonic.scripts.run_h2_bvh_reference_batch import ROOT,BUNDLE,digest,atomic_json,verify_bundle,compact_trace,seal_jobs,raw_valid


def jobs_for(out):
    return [job for p in sorted((out/'work').glob('gpu*_batch*/jobs.json')) for job in json.loads(p.read_text())]


def migrate(a):
    import numpy as np
    signature=verify_bundle();old=json.loads((a.source/'manifest.json').read_text())
    # A new bundle may change orchestration and the GPU solver, never silently calibration.
    old_files=json.loads((Path(old['bundle'])/'bundle_manifest.json').read_text())['files']
    new_files=json.loads((BUNDLE/'bundle_manifest.json').read_text())['files']
    for rel,sha in old_files.items():
        if rel.startswith(('profile/','soma-retargeter-h2/','code/gear_sonic/data/')) and new_files.get(rel)!=sha:
            raise ValueError('Calibration/model/source retarget code changed: '+rel)
    if not a.out.exists():a.out.mkdir(parents=True)
    lock=(a.source/'launch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    dest=a.out/'manifest.json'
    if dest.exists():
        current=json.loads(dest.read_text())
        if current['bundle_sha256']!=signature or current['motions']!=old['motions']:raise ValueError('Incompatible destination manifest')
    else:
        old.update(schema='h2_bvh_gpu_staged.v3',bundle=str(BUNDLE),bundle_sha256=signature,previous_run=str(a.source),previous_manifest_sha256=digest(a.source/'manifest.json'),scope=__doc__,training_started=False)
        atomic_json(dest,old)
    if (a.out/'migration.json').exists():
        print((a.out/'migration.json').read_text(),flush=True);return
    copied=0
    for src in sorted((a.source/'work').glob('gpu*_batch*/jobs.json')):
        work=a.out/'work'/src.parent.name;work.mkdir(parents=True,exist_ok=True);jobs=[]
        for oldjob in json.loads(src.read_text()):
            job=dict(oldjob);key=job['key'];job.update(work=str(work),out=str(a.out),reference=str(BUNDLE/'profile/reference.npz'))
            if digest(job['bvh'])!=job['bvh_sha256'] or digest(job['g1_csv'])!=job['g1_csv_sha256']:raise ValueError('Source changed: '+key)
            for robot in ('h2','g1'):
                for rel in (Path(robot)/'csv'/f'{key}.csv',):
                    dst=work/rel;dst.parent.mkdir(parents=True,exist_ok=True)
                    if not dst.exists():shutil.copyfile(src.parent/rel,dst)
                rel=Path(robot)/f'{key}_trace.npz';dst=work/rel
                if not dst.exists():
                    with np.load(src.parent/rel) as z:trace={k:z[k] for k in z.files}
                    trace['source_world'],names=compact_trace(trace['source_world'],trace['source_names'].tolist());trace['source_names']=np.array(names)
                    with dst.with_suffix('.partial').open('wb') as f:np.savez_compressed(f,**trace)
                    dst.with_suffix('.partial').replace(dst)
            jobs.append(job);copied+=1
        seal_jobs(work,jobs)
    atomic_json(a.out/'migration.json',dict(previous_run=str(a.source),raw_motions_reused=copied,previous_reports_preserved=len(list((a.source/'reports').glob('*.json'))),previous_files_deleted=False))
    print(json.dumps(dict(raw_motions_reused=copied,out=str(a.out))),flush=True)


def bind_run_settings(out, settings):
    """Prevent mixing outputs from different numerical settings on resume."""
    path=out/'gpu_settings.json'
    if path.exists():
        if json.loads(path.read_text())!=settings:
            raise ValueError('GPU settings changed; use a new output directory')
    else:
        if any((out/'refined/reports').glob('*.json')):
            raise ValueError('Existing refinement has no settings receipt')
        atomic_json(path,settings)


def run_child(command, env):
    proc=subprocess.Popen(command,cwd=ROOT,env=env,start_new_session=True)
    try:
        code=proc.wait()
        if code:raise subprocess.CalledProcessError(code,command)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL);proc.wait()


def refine_worker(a):
    import numpy as np,torch
    from scipy.spatial.transform import Rotation as R
    from gear_sonic.scripts.h2_bvh_reference_worker import read_csv,write_csv
    from gear_sonic.scripts.h2_gpu_reference_ik import GPUReferenceIK
    if not (a.out/'raw_complete.json').exists():raise ValueError('All raw retargets must finish first')
    if verify_bundle()!=json.loads((a.out/'manifest.json').read_text())['bundle_sha256']:raise ValueError('Bundle mismatch')
    settings=dict(frame_chunk=a.frame_chunk,iterations=a.iterations,bundle_sha256=verify_bundle())
    if json.loads((a.out/'gpu_settings.json').read_text())!=settings:raise ValueError('Worker settings mismatch')
    solver=GPUReferenceIK(ROOT/'gear_sonic/data/assets/robot_description/mjcf/h2.xml')
    jobs=jobs_for(a.out)[a.shard::2];completed=0;reused=0;started=time.time();group=[];frames=0
    atomic_json(a.out/f'refine_{a.shard}.json',dict(stage='gpu_refinement',completed=0,reused=0,total=len(jobs)))
    def flush():
        nonlocal completed,group,frames
        if not group:return
        start=time.time();starts=np.cumsum([0]+[len(g['q']) for g in group])[:-1].tolist();tn=group[0]['tn'];fps=group[0]['fps']
        q=np.concatenate([g['q'] for g in group]);targets=np.concatenate([g['targets'] for g in group]);directions={s:np.concatenate([g['directions'][s] for g in group]) for s in ('Left','Right')}
        with np.load(BUNDLE/'profile/reference.npz') as ref:canonical=ref['qpos']
        q=solver.solve_chunk(q,targets,tn,directions,a.iterations,canonical)
        histories={}
        for side in ('Left','Right'):
            wrist=R.from_quat(targets[:,tn.index(side+'Hand'),3:])
            q,histories[side]=solver.sequence_arm(q,side,directions[side],wrist.as_matrix(),fps,a.iterations,starts)
            q=solver.resample_wrists(q,side,wrist,fps,50.,.15,a.iterations,starts)
        torch.cuda.synchronize()
        for g,begin in zip(group,starts):
            job=g['job'];key=job['key'];result=q[begin:begin+len(g['q'])];raw=g['raw'].copy();raw[:,1:4]=result[:,:3];raw[:,4:7]=R.from_quat(result[:,[4,5,6,3]]).as_euler('xyz');raw[:,g['cols']]=result[:,7:]
            if not np.array_equal(raw[:,0],g['raw'][:,0]):raise ValueError('Timebase changed')
            dest=a.out/'refined/csv'/f'{key}.csv';write_csv(dest.with_suffix('.partial'),g['header'],raw);dest.with_suffix('.partial').replace(dest)
            atomic_json(a.out/'refined/reports'/f'{key}.json',dict(key=key,output=str(dest),output_sha256=digest(dest),raw_files=job['raw_files'],bundle_sha256=json.loads((a.out/'manifest.json').read_text())['bundle_sha256'],frames=len(raw),fps=fps,backend='CUDA',group_motions=len(group),group_frames=len(q),group_seconds=time.time()-start,temporal_history=histories,admitted=False,training_started=False))
            completed+=1
        atomic_json(a.out/f'refine_{a.shard}.json',dict(stage='gpu_refinement',completed=completed,reused=reused,total=len(jobs),seconds=time.time()-started))
        print(json.dumps(dict(shard=a.shard,completed=completed,total=len(jobs),group_motions=len(group),group_frames=len(q),seconds=round(time.time()-start,2))),flush=True)
        group=[];frames=0
    for job in jobs:
        key=job['key'];report=a.out/'refined/reports'/f'{key}.json'
        if report.exists():
            r=json.loads(report.read_text())
            if r.get('bundle_sha256')==settings['bundle_sha256'] and r.get('raw_files')==job['raw_files'] and Path(r['output']).is_file() and digest(r['output'])==r['output_sha256']:
                completed+=1;reused+=1;continue
        work=Path(job['work'])
        if not raw_valid(work,job):raise ValueError('Invalid raw input: '+key)
        header,raw=read_csv(work/'h2/csv'/f'{key}.csv');clean=[n.removesuffix('_dof') for n in header];cols=[clean.index(n) for n in solver.names]
        with np.load(work/'h2'/f'{key}_trace.npz') as z:t={k:z[k] for k in z.files}
        sn=t['source_names'].tolist();tn=t['names'].tolist();fps=float(t['fps']);directions={}
        if len(raw)!=len(t['targets']) or not np.isfinite(raw).all():raise ValueError('Invalid frame data: '+key)
        if group and (frames+len(raw)>a.frame_chunk or tn!=group[0]['tn'] or fps!=group[0]['fps']):flush()
        for side in ('Left','Right'):
            v=np.diff(t['source_world'][:,[sn.index(side+s) for s in ('Arm','ForeArm','Hand')],:3],axis=1);norm=np.linalg.norm(v,axis=-1,keepdims=True)
            if np.any(norm<1e-8):raise ValueError('Degenerate human arm segment: '+key)
            directions[side]=v/norm
        q=np.c_[raw[:,1:4],R.from_euler('xyz',raw[:,4:7]).as_quat()[:,[3,0,1,2]],raw[:,cols]]
        group.append(dict(job=job,q=q,header=header,raw=raw,cols=cols,targets=t['targets'],tn=tn,fps=fps,directions=directions));frames+=len(raw)
    flush();atomic_json(a.out/f'refine_{a.shard}.json',dict(stage='complete',completed=completed,reused=reused,total=len(jobs),seconds=time.time()-started))


def audit_one(job):
    from gear_sonic.scripts.h2_bvh_reference_worker import compare,ARM_METRICS,FOOT_METRICS
    from gear_sonic.scripts.audit_g1_style_retarget import evaluate
    out=Path(job['out']);work=Path(job['work']);key=job['key'];path=out/'refined/csv'/f'{key}.csv';reference=Path(job['reference'])
    started=time.time();h={};g={};checks=[];physical=True
    for rate in (50,120):
        h[rate]=evaluate(path,work/'h2'/f'{key}_trace.npz','h2',reference,rate);g[rate]=evaluate(work/'g1/csv'/f'{key}.csv',work/'g1'/f'{key}_trace.npz','g1',reference,rate)
        for side in ('Left','Right'):
            for section,metrics in [('arms',ARM_METRICS),('feet',FOOT_METRICS)]:
                checks.extend(dict(fps=rate,side=side,section=section,metric=k,stat=stat,h2=x,g1=y,passed=ok) for k,stat,x,y,ok in compare(h[rate][section][side],g[rate][section][side],metrics))
        checks.extend(dict(fps=rate,section='torso',metric=k,stat=stat,h2=x,g1=y,passed=ok) for k,stat,x,y,ok in compare(h[rate],g[rate],('torso_lateral_error_deg','torso_orientation_error_deg')))
        c=h[rate]['native_checks'];physical &= c['joint_limit_excess_deg']<=1e-4 and c['max_urdf_velocity_ratio']<=1+1e-6 and c['shoulder_wrist_min_limit_margin_deg']>=2-1e-6
    status='accepted' if physical and all(c['passed'] for c in checks) else 'needs_review';dest=out/status/'csv'/f'{key}.csv';dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():
        if digest(dest)!=digest(path):raise ValueError('Existing admission output differs: '+key)
    else:os.link(path,dest)
    atomic_json(out/'reports'/f'{key}.json',dict(key=key,status=status,output=str(dest),output_sha256=digest(dest),refined_sha256=digest(path),raw_files=job['raw_files'],checks=checks,physical_and_margin_pass=bool(physical),audit_h2=h,audit_g1=g,seconds=time.time()-started,source_bvh_sha256=job['bvh_sha256'],g1_retarget=False,training_started=False))
    return status


def audit(a):
    if a.stop_after=='raw':return
    if not (a.out/'refine_complete.json').exists():raise ValueError('GPU refinement must finish before audit')
    counts=dict(accepted=0,needs_review=0,error=0);jobs=jobs_for(a.out)
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.cpu_workers) as pool:
        futures={}
        for job in jobs:
            p=a.out/'reports'/f"{job['key']}.json"
            if p.exists():
                r=json.loads(p.read_text())
                if r['status'] in ('accepted','needs_review') and r.get('raw_files')==job['raw_files'] and r.get('refined_sha256')==digest(a.out/'refined/csv'/f"{job['key']}.csv") and Path(r['output']).is_file() and digest(r['output'])==r['output_sha256']:counts[r['status']]+=1;continue
            futures[pool.submit(audit_one,job)]=job['key']
        for f in concurrent.futures.as_completed(futures):
            try:counts[f.result()]+=1
            except Exception as e:
                counts['error']+=1;atomic_json(a.out/'reports'/f'{futures[f]}.json',dict(key=futures[f],status='error',error=repr(e)))
            progress=dict(stage='final_audit',completed=sum(counts.values()),total=len(jobs),counts=counts,training_started=False)
            atomic_json(a.out/'audit_progress.json',progress);print(json.dumps(progress),flush=True)
    final=dict(stage='complete',completed=sum(counts.values()),total=len(jobs),counts=counts,training_started=False)
    if counts['error']:raise RuntimeError('Audit errors; inspect reports')
    atomic_json(a.out/'complete.json',final)


def run(a):
    signature=verify_bundle();manifest=json.loads((a.out/'manifest.json').read_text())
    if manifest['bundle_sha256']!=signature:raise ValueError('Bundle mismatch')
    lock=(a.out/'staged.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    bind_run_settings(a.out,dict(frame_chunk=a.frame_chunk,iterations=a.iterations,bundle_sha256=signature))
    (a.out/'logs').mkdir(exist_ok=True)
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH=str(ROOT)+os.pathsep+str(BUNDLE/'soma-retargeter-h2'),SOMA_RETARGETER_ROOT=str(BUNDLE/'soma-retargeter-h2'))
    if not (a.out/'raw_complete.json').exists():
        run_child([sys.executable,'-u','-m','gear_sonic.scripts.run_h2_bvh_reference_batch','launch','--raw-only','--out',str(a.out),'--gpus','0,1','--batch-size-total','512'],env)
    jobs=jobs_for(a.out)
    if len(jobs)!=len(manifest['motions']) or len({j['key'] for j in jobs})!=len(jobs):raise ValueError('Raw dataset incomplete or duplicate')
    if a.stop_after=='raw':return
    if not (a.out/'refine_complete.json').exists():
        children=[]
        try:
            for shard in (0,1):
                cmd=[sys.executable,'-u','-m','gear_sonic.scripts.run_h2_bvh_gpu_staged','refine_worker','--out',str(a.out),'--shard',str(shard),'--frame-chunk',str(a.frame_chunk),'--iterations',str(a.iterations)]
                log=(a.out/'logs'/f'refine_{shard}.log').open('a');proc=subprocess.Popen(cmd,cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=str(shard)),stdout=log,stderr=subprocess.STDOUT,start_new_session=True);children.append((proc,log))
            while any(p.poll() is None for p,l in children):
                if any(p.poll() not in (None,0) for p,l in children):raise RuntimeError('GPU refinement failed; inspect refine logs')
                progress=dict(stage='gpu_refinement',shards=[json.loads(p.read_text()) for p in [a.out/'refine_0.json',a.out/'refine_1.json'] if p.exists()],training_started=False);atomic_json(a.out/'refine_progress.json',progress);print(json.dumps(progress),flush=True);time.sleep(5)
            if any(p.returncode for p,l in children):raise RuntimeError('GPU refinement failed')
        finally:
            for p,l in children:
                if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
            for p,l in children:p.wait();l.close()
        records=[json.loads((a.out/f'refine_{s}.json').read_text()) for s in (0,1)]
        if sum(r['completed'] for r in records)!=len(jobs):raise RuntimeError('Incomplete refinement')
        atomic_json(a.out/'refine_complete.json',dict(completed=len(jobs),backend='CUDA',training_started=False))
    if a.stop_after=='refine':return
    audit(a)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['migrate','run','refine_worker','audit']);p.add_argument('--out',type=Path,required=True);p.add_argument('--source',type=Path);p.add_argument('--shard',type=int,default=0);p.add_argument('--frame-chunk',type=int,default=8192);p.add_argument('--iterations',type=int,default=100);p.add_argument('--cpu-workers',type=int,default=32);p.add_argument('--stop-after',choices=['raw','refine','audit'],default='audit');a=p.parse_args()
    if a.iterations<1 or a.frame_chunk<1 or a.cpu_workers<1:p.error('Counts must be positive')
    def interrupted(signum,frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,interrupted)
    a.out=a.out.resolve()
    if a.mode=='migrate' and not a.source:p.error('--source required')
    globals()[a.mode](a)
if __name__=='__main__':main()
