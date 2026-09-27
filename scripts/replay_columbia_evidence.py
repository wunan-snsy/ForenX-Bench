#!/usr/bin/env python3
"""Independent replay using only frozen inputs, evidence JSONL and model artifact."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from run_columbia_blocks_baseline import REPO, file_hash, sigmoid
from run_columbia_full_stress import descriptor

def main():
 cfg=json.loads((REPO/'configs/evidence_replay.json').read_text()); base=REPO/'results/columbia_evidence'; root=REPO/'data/external/columbia_uncompressed'
 items=[json.loads(x) for x in (base/'evidence.jsonl').read_text().splitlines() if x.strip()]; model=json.loads((base/'model.json').read_text())
 complete=replayed=agreed=0; failures=[]
 for e in items:
  missing=[k for k in cfg['required_fields'] if k not in e]
  if not missing: complete+=1
  else: failures.append({'id':e.get('evidence_id'),'type':'traceability','missing':missing}); continue
  p=root/e['input']['relative_path']
  try:
   if file_hash(p)!=e['input']['sha256']: raise ValueError('input_hash_mismatch')
   if file_hash(base/e['model']['artifact'])!=e['model']['sha256']: raise ValueError('model_hash_mismatch')
   z=descriptor(Image.open(p)); mu=np.array(model['mean']); sd=np.array(model['scale']); w=np.array(model['weights'])
   score=float(sigmoid(np.r_[1.,(z-mu)/sd]@w)); replayed+=1
   same_score=abs(score-e['observation']['score_spliced'])<=cfg['score_tolerance']; pred='spliced' if score>=model['threshold'] else 'authentic'
   if same_score and pred==e['observation']['predicted_class']: agreed+=1
   else: failures.append({'id':e['evidence_id'],'type':'agreement','score_delta':score-e['observation']['score_spliced'],'replayed_class':pred})
  except Exception as ex: failures.append({'id':e['evidence_id'],'type':'replay','error':str(ex)})
 n=len(items); result={'status':'complete','evidence_items':n,'traceable_items':complete,'replayed_items':replayed,'agreed_items':agreed,'ETR':complete/n if n else None,'RSR':replayed/n if n else None,'EGA':agreed/replayed if replayed else None,'failures':failures}
 (base/'replay.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
if __name__=='__main__':main()
