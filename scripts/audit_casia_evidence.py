#!/usr/bin/env python3
"""Audit CASIA evidence completeness and verify hashes against source artifacts."""
import csv,hashlib,json,sys
from collections import Counter
from pathlib import Path
import numpy as np
from run_columbia_blocks_baseline import REPO
from evaluate_casia_v1_models import prediction_path

EVIDENCE=REPO/'results/casia_v1_evidence/evidence.jsonl'
OUT=REPO/'results/casia_v1_evidence/audit_v1_1.json'
REQUIRED={'schema_version','evidence_id','model_id','sample_id','class_reference','input','claim','observation','operation','applicability','alternatives','failure_modes','uncertainty'}

def sha_file(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def sha_map(array):
 a=np.ascontiguousarray(array)
 header=json.dumps({'encoding':'numpy-contiguous-v1','dtype':a.dtype.str,'shape':list(a.shape)},separators=(',',':')).encode()
 return hashlib.sha256(header+b'\0'+a.tobytes()).hexdigest()

def main():
 manifest=list(csv.DictReader((REPO/'manifests/casia_v1_manifest.csv').open(encoding='utf-8')))
 rows={(m,s['sample_id']):s for m in ('trufor','mvss') for s in manifest}
 seen=Counter(); errors=[]; complete=0; hash_ok=0; consistency_ok=0
 data_root=REPO/'data/external/casia_v1_corrected'
 for lineno,line in enumerate(EVIDENCE.open(encoding='utf-8'),1):
  try:item=json.loads(line)
  except Exception as e:errors.append({'line':lineno,'error':'invalid_json','detail':str(e)});continue
  model=item.get('model_id');sid=item.get('sample_id');key=(model,sid);seen[key]+=1
  missing=sorted(REQUIRED-set(item))
  if missing:errors.append({'line':lineno,'sample_id':sid,'error':'missing_fields','fields':missing});continue
  obs=item.get('observation',{});inp=item.get('input',{});op=item.get('operation',{})
  if item.get('schema_version')!='forenx-evidence-1.1':errors.append({'line':lineno,'sample_id':sid,'error':'wrong_schema_version'})
  required_nested=[('input',('relative_path','sha256','manifest_sha256','hash_verified')),('observation',('image_score','threshold','decision','map_sha256','prediction_artifact_sha256','map_hash_encoding')),('operation',('id','parameters'))]
  nested_missing=[f'{obj}.{key2}' for obj,keys in required_nested for key2 in keys if key2 not in item.get(obj,{})]
  if nested_missing:errors.append({'line':lineno,'sample_id':sid,'error':'missing_nested_fields','fields':nested_missing});continue
  src=rows.get(key)
  if src is None:errors.append({'line':lineno,'sample_id':sid,'error':'unknown_model_or_sample'});continue
  complete+=1
  input_path=data_root/inp['relative_path'];pred=prediction_path(model,src)
  actual_input=sha_file(input_path) if input_path.is_file() else None
  input_match=actual_input==src['sha256']==inp['sha256']==inp['manifest_sha256'] and inp['hash_verified'] is True
  z=np.load(pred);map_match=sha_map(z['map'])==obs['map_sha256'];pred_match=sha_file(pred)==obs['prediction_artifact_sha256']
  if input_match and map_match and pred_match:hash_ok+=1
  else:errors.append({'line':lineno,'sample_id':sid,'error':'integrity_mismatch','input_match':input_match,'map_hash_match':map_match,'prediction_artifact_hash_match':pred_match})
  score_match=abs(float(z['score'])-float(obs['image_score']))<=1e-8
  decision=('tampered' if float(obs['image_score'])>=float(obs['threshold']) else 'authentic')==obs['decision']
  class_match=src['class']==item['class_reference']
  if score_match and decision and class_match:consistency_ok+=1
  else:errors.append({'line':lineno,'sample_id':sid,'error':'observation_consistency_mismatch','score_match':score_match,'decision_match':decision,'class_match':class_match})
 expected=len(rows);duplicates=[{'model':m,'sample_id':s,'count':n} for (m,s),n in seen.items() if n>1]
 missing_records=[{'model':m,'sample_id':s} for m,s in rows if seen[(m,s)]==0]
 report={'status':'complete' if not errors and not duplicates and not missing_records and len(seen)==expected else 'audit_failed','schema_version':'forenx-evidence-1.1','expected_decisions':expected,'records':sum(seen.values()),'ECR':len(seen.keys() & rows.keys())/expected,'complete_records':complete,'ETR':complete/max(1,sum(seen.values())),'integrity_verified_records':hash_ok,'integrity_verification_rate':hash_ok/max(1,complete),'observation_consistent_records':consistency_ok,'observation_consistency_rate':consistency_ok/max(1,complete),'duplicates':duplicates,'missing_records':missing_records,'errors':errors}
 OUT.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
