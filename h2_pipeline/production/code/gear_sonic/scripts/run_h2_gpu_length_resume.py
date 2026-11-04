"""Import sealed raw results and length-sort only unfinished H2 BVH motions."""
import argparse,concurrent.futures,contextlib,fcntl,hashlib,json,os,re,time
from pathlib import Path
from gear_sonic.scripts.run_h2_bvh_reference_batch import BUNDLE,digest,atomic_json,verify_bundle,raw_valid


def frame_count(path):
    with Path(path).open('rb') as f:header=f.read(65536)
    match=re.search(rb'(?m)^Frames:\s*(\d+)\s*\r?$',header)
    if not match or int(match.group(1))<1:raise ValueError('Missing BVH frame count: '+str(path))
    return int(match.group(1))


def length_plan(motions,completed,lengths,batch=256):
    keys=[m['key'] for m in motions]
    if len(set(keys))!=len(keys) or not set(completed)<=set(keys):raise ValueError('Duplicate or unknown motion keys')
    shards=[];old_steps=0;new_steps=0
    for shard in (0,1):
        pending=[k for k in keys[shard::2] if k not in completed]
        old_steps+=sum(max(lengths[k]+15 for k in pending[i:i+batch]) for i in range(0,len(pending),batch))
        ordered=sorted(pending,key=lambda k:(-lengths[k],k))
        new_steps+=sum(max(lengths[k]+15 for k in ordered[i:i+batch]) for i in range(0,len(ordered),batch))
        shards.append(ordered)
    return dict(schema='h2_length_resume.v1',batch_size_per_gpu=batch,shards=shards,
                imported_keys=sorted(completed),remaining=len(keys)-len(completed),
                pending_frame_sum=sum(lengths[k] for s in shards for k in s),
                unsorted_batch_max_frame_sum=old_steps,sorted_batch_max_frame_sum=new_steps,
                note='Frame-loop proxy only; excludes skeleton subgroups, preparation and IO; not a speed guarantee')


def active_source_processes(source):
    modules={'gear_sonic.scripts.run_h2_bvh_gpu_staged','gear_sonic.scripts.run_h2_bvh_reference_batch'}
    found=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdecimal():continue
        try:args=[v.decode(errors='replace') for v in (p/'cmdline').read_bytes().split(b'\0') if v]
        except OSError:continue
        if modules.intersection(args) and str(source) in args:found.append(int(p.name))
    return found


def compatible_source(source_manifest):
    old_bundle=Path(source_manifest['bundle'])
    if digest(old_bundle/'bundle_manifest.json')!=source_manifest['bundle_sha256']:raise ValueError('Old bundle manifest changed')
    old=json.loads((old_bundle/'bundle_manifest.json').read_text())['files']
    new=json.loads((BUNDLE/'bundle_manifest.json').read_text())['files']
    allowed={'soma-retargeter-h2/soma_retargeter/pipelines/newton_pipeline.py',
             'soma-retargeter-h2/soma_retargeter/pipelines/feet_stabilizer.py'}
    for rel,sha in old.items():
        if rel.startswith(('profile/','approved/','soma-retargeter-h2/','code/gear_sonic/data/')) and new.get(rel)!=sha and rel not in allowed:
            raise ValueError('Calibration or model changed: '+rel)
    proof=json.loads((BUNDLE/'transfer_equivalence.json').read_text())
    if not proof['passed'] or proof['baseline_bundle_sha256']!=source_manifest['bundle_sha256']:
        raise ValueError('Missing equivalence evidence for source bundle')
    for rel,sha in proof['tested_files'].items():
        if new.get(rel)!=sha:raise ValueError('Transfer implementation differs from tested code: '+rel)


def prepare(source,out):
    with contextlib.ExitStack() as stack:
        return _prepare(source,out,stack)


def _prepare(source,out,stack):
    signature=verify_bundle();source=source.resolve();out=out.resolve()
    if source==out:raise ValueError('Use a separate resume directory')
    old=json.loads((source/'manifest.json').read_text());compatible_source(old)
    locks=[]
    for name in ('staged.lock','launch.lock'):
        lock=stack.enter_context((source/name).open('a'))
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Previous run is active; run stop_v4.sh first')
        locks.append(lock)
    live=active_source_processes(source)
    if live:raise RuntimeError('Previous workers still running: '+str(live))
    out.mkdir(parents=True,exist_ok=True)
    own=stack.enter_context((out/'prepare.lock').open('a'));fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    receipt=out/'resume_import.json'
    if receipt.exists():
        r=json.loads(receipt.read_text())
        if r['bundle_sha256']!=signature or r['previous_manifest_sha256']!=digest(source/'manifest.json'):raise ValueError('Resume identity mismatch')
        current=json.loads((out/'manifest.json').read_text())
        if current['bundle_sha256']!=signature or digest(out/'length_plan.json')!=current['length_plan_sha256']:raise ValueError('Resume plan changed')
        print(json.dumps(r),flush=True);return
    if (out/'raw_complete.json').exists() or any((out/'work').glob('gpu[01]_batch*/jobs.json')):
        raise ValueError('Unsealed import has already generated new work')
    motion_by_key={m['key']:m for m in old['motions']};completed=set();started=time.time()
    sources=sorted((source/'work').glob('gpu*_batch*/jobs.json'))
    def link_job(pair):
        job,work=pair;key=job['key'];item=motion_by_key[key]
        if job['bvh']!=item['bvh'] or job['g1_csv']!=item['g1_csv']:raise ValueError('Source mismatch: '+key)
        for field,size,mtime in [('bvh','size','mtime_ns'),('g1_csv','g1_size','g1_mtime_ns')]:
            st=Path(item[field]).stat()
            if (st.st_size,st.st_mtime_ns)!=(item[size],item[mtime]):raise ValueError('Input changed: '+key)
        src=Path(job['work'])
        if not raw_valid(src,job):raise ValueError('Sealed source raw files corrupt: '+key)
        for rel,sha in job['raw_files'].items():
            target=work/rel;target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():os.link(src/rel,target)
            elif os.stat(target).st_ino!=os.stat(src/rel).st_ino or os.stat(target).st_dev!=os.stat(src/rel).st_dev:
                if digest(target)!=sha:raise ValueError('Existing import differs: '+key)
        return dict(job,work=str(work),out=str(out),reference=str(BUNDLE/'profile/reference.npz'),origin_bundle_sha256=old['bundle_sha256'])
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for i,path in enumerate(sources):
            jobs=json.loads(path.read_text());work=out/'work'/f'gpu_reuse_batch{i:06d}'
            for job in jobs:
                if job['key'] in completed or job['key'] not in motion_by_key:raise ValueError('Duplicate or unknown sealed key')
                completed.add(job['key'])
            imported=list(pool.map(link_job,[(j,work) for j in jobs]));atomic_json(work/'jobs.json',imported)
            print(json.dumps(dict(stage='import_verified_raw',completed=len(completed),source_batches_done=i+1,source_batches=len(sources),seconds=round(time.time()-started,1))),flush=True)
        pending=[m for m in old['motions'] if m['key'] not in completed]
        inventory=BUNDLE/'length_inventory.json'
        if inventory.exists():
            cache=json.loads(inventory.read_text())
            if cache['motions_sha256']==hashlib.sha256(json.dumps(old['motions'],sort_keys=True).encode()).hexdigest():
                lengths={m['key']:cache['frames'][m['key']] for m in pending}
            else:lengths={m['key']:n for m,n in zip(pending,pool.map(frame_count,[m['bvh'] for m in pending]))}
        else:lengths={m['key']:n for m,n in zip(pending,pool.map(frame_count,[m['bvh'] for m in pending]))}
    plan=length_plan(old['motions'],completed,lengths);atomic_json(out/'length_plan.json',plan)
    manifest=dict(old,schema='h2_gpu_length_resume.v5',bundle=str(BUNDLE),bundle_sha256=signature,
                  previous_run=str(source),previous_manifest_sha256=digest(source/'manifest.json'),
                  length_plan_sha256=digest(out/'length_plan.json'),gpu_transfer_batching=True,training_started=False)
    atomic_json(out/'manifest.json',manifest)
    record=dict(bundle_sha256=signature,previous_manifest_sha256=digest(source/'manifest.json'),
                previous_run=str(source),reused=len(completed),remaining=plan['remaining'],
                raw_files_reused_by='verified_hardlinks',old_files_deleted=False,
                unsorted_batch_max_frame_sum=plan['unsorted_batch_max_frame_sum'],sorted_batch_max_frame_sum=plan['sorted_batch_max_frame_sum'],
                previous_refinement_reports_preserved=len(list((source/'refined/reports').glob('*.json'))),seconds=time.time()-started)
    atomic_json(receipt,record);print(json.dumps(record),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();prepare(a.source,a.out)
if __name__=='__main__':main()
