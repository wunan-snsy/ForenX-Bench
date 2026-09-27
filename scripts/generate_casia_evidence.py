#!/usr/bin/env python3
"""Create machine-readable evidence packages for all CASIA model decisions."""
import csv, hashlib, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from run_columbia_blocks_baseline import REPO
from evaluate_casia_v1_models import prediction_path

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''): h.update(chunk)
    return h.hexdigest()

def sha_map(array):
    """Hash canonical map values plus dtype and shape, not the enclosing NPZ."""
    a=np.ascontiguousarray(array)
    header=json.dumps({'encoding':'numpy-contiguous-v1','dtype':a.dtype.str,'shape':list(a.shape)},separators=(',',':')).encode('utf-8')
    return hashlib.sha256(header+b'\0'+a.tobytes()).hexdigest()

def main():
    with (REPO/'manifests/casia_v1_manifest.csv').open(encoding='utf-8') as f: rows=list(csv.DictReader(f))
    root=REPO/'data/external/casia_v1_corrected'; out=REPO/'results/casia_v1_evidence';out.mkdir(parents=True,exist_ok=True)
    all_items=[]; missing=[]
    meta={'trufor':{'code_commit':'ae54475df6f41a491d7615100feb19263dec13f7','weights_md5':'7bee48f3476c75616c3c5721ab256ff8'},
          'mvss':{'weights_sha256':'080bc6c3aae59f748b547dbf090786fe9d31a6e50749daaa40871e298d6a7e50','preprocessing':'official 512x512 BGR ImageNet normalization'}}
    for model in meta:
      for row in rows:
        image=root/row['relative_path']; pred=prediction_path(model,row)
        try:
          actual_input_sha=sha(image)
          if actual_input_sha!=row['sha256']:
            raise ValueError('input hash differs from manifest')
          z=np.load(pred); score=float(z['score']); decision='tampered' if score>=.5 else 'authentic'
          item={'schema_version':'forenx-evidence-1.1','evidence_id':hashlib.sha256((model+row['sha256']).encode()).hexdigest(),
            'model_id':model,'sample_id':row['sample_id'],'class_reference':row['class'],
            'input':{'relative_path':row['relative_path'],'sha256':actual_input_sha,'manifest_sha256':row['sha256'],'hash_verified':True},
            'claim':f'{model} score is {score:.6f}; fixed-threshold decision is {decision}.',
            'observation':{'image_score':score,'threshold':.5,'decision':decision,'map_sha256':sha_map(z['map']),'map_hash_encoding':'numpy-contiguous-v1','prediction_artifact_sha256':sha(pred)},
            'operation':{'id':f'{model}.frozen_inference.v1','parameters':meta[model]},
            'applicability':['CASIA v1 corrected image','frozen published model artifact'],
            'alternatives':['compression or resampling artifact','dataset-specific acquisition characteristic'],
            'failure_modes':['distribution shift','threshold miscalibration','map semantics differ from reference mask'],
            'uncertainty':'Model score is not an empirical probability of tampering.'}
          all_items.append(item)
        except Exception as e: missing.append({'model':model,'sample_id':row['sample_id'],'error':str(e)})
    (out/'evidence.jsonl').write_text('\n'.join(json.dumps(x,separators=(',',':')) for x in all_items)+'\n',encoding='utf-8')
    result={'status':'complete' if not missing else 'incomplete','created_utc':datetime.now(timezone.utc).isoformat(),
      'schema_version':'forenx-evidence-1.1','expected_decisions':len(rows)*2,'evidence_items':len(all_items),
      'ECR':len(all_items)/(len(rows)*2),'ETR':None if missing else 'computed by scripts/audit_casia_evidence.py after structural and hash verification','failures':missing}
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
