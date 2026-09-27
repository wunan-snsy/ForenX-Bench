#!/usr/bin/env python3
import json,subprocess
from pathlib import Path
import numpy as np
from run_columbia_blocks_baseline import REPO,file_hash

def main():
 cfg=json.loads((REPO/'configs/trufor_evidence_replay.json').read_text()); items=[json.loads(x) for x in (REPO/'results/trufor_spatial_evidence/evidence.jsonl').read_text().splitlines()]
 chosen={}
 for e in items:
  pair='_'.join(e['sample_id'].split('_')[:2]); chosen.setdefault(pair,e)
 audit=[chosen[k] for k in sorted(chosen)]; out=REPO/'results/trufor_spatial_replay';out.mkdir(parents=True,exist_ok=True)
 py=REPO/'external/TruFor/.venv/Scripts/python.exe'; script=REPO/'external/TruFor/test_docker/src/trufor_test.py'; cwd=script.parent; records=[]
 for e in audit:
  inp=REPO/e['input']['relative_path']; target=out/(e['sample_id']+'.npz')
  if target.exists():target.unlink()
  cp=subprocess.run([str(py),str(script),'-gpu','0','-in',str(inp),'-out',str(target)],cwd=cwd,text=True,capture_output=True)
  rec={'sample_id':e['sample_id'],'exit_code':cp.returncode}
  try:
   if cp.returncode:raise RuntimeError(cp.stderr[-500:])
   old=np.load(REPO/'results/trufor_columbia_spliced'/(inp.name+'.npz')); new=np.load(target)
   rec.update({'score_abs_delta':abs(float(old['score'])-float(new['score'])),'map_max_abs_delta':float(np.max(abs(old['map']-new['map']))),'confidence_max_abs_delta':float(np.max(abs(old['conf']-new['conf'])))})
   rec['agree']=rec['score_abs_delta']<=cfg['score_tolerance'] and rec['map_max_abs_delta']<=cfg['map_max_abs_tolerance'] and rec['confidence_max_abs_delta']<=cfg['confidence_max_abs_tolerance']
  except Exception as ex:rec.update({'agree':False,'error':str(ex)})
  records.append(rec); print(json.dumps(rec))
 successful=sum(r['exit_code']==0 for r in records); agreed=sum(r['agree'] for r in records)
 result={'status':'complete','selection':cfg['audit_selection'],'audit_items':len(records),'successful_replays':successful,'agreed_replays':agreed,'RSR':successful/len(records),'EGA':agreed/successful if successful else None,'records':records}
 (out/'replay.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
