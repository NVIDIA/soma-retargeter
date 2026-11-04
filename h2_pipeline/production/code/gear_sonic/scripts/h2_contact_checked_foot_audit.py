"""Experimental foot guard plus explicit review of new ground-crossing events."""
from pathlib import Path
import importlib.util,os
spec=importlib.util.spec_from_file_location('expanded_foot_guard',Path(__file__).with_name('h2_experimental_foot_audit.py'));foot=importlib.util.module_from_spec(spec);spec.loader.exec_module(foot)

def requires_contact_review(report):
 guard=report['trajectory_guard']
 if not guard['passed']:return False
 rates=guard.get('clearance_diagnostics',{}).get('rates',{})
 if not rates:raise ValueError('Missing clearance diagnostics')
 return any(v['zero_clearance_crossing_frames']>0 for v in rates.values())

def audit_task(task):
 key,status=foot.audit_task(task);out=Path(task[2]);path=out/'reports'/f'{key}.json';r=foot.base.read(path);review=requires_contact_review(r)
 r['contact_review_required']=review;r['contact_screen']='baseline/candidate sole zero-clearance crossing; FK proxy only'
 if status=='accepted' and review:
  dest=out/'needs_review/csv'/f'{key}.csv';dest.parent.mkdir(parents=True,exist_ok=True);old=Path(r['output'])
  if dest.exists():
   if foot.base.digest(dest)!=r['output_sha256']:raise ValueError('Changed review CSV')
   old.unlink()
  else:old.rename(dest)
  r['status']='needs_review';r['output']=str(dest);status='needs_review'
 foot.base.save(path,r);return key,status
