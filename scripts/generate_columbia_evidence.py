#!/usr/bin/env python3
"""Generate bounded, machine-readable evidence packages for the frozen full-image pilot."""
import hashlib, json
from datetime import datetime, timezone
import numpy as np
from PIL import Image
from run_columbia_blocks_baseline import REPO, fit, predict, threshold_dev, file_hash
from run_columbia_full_detection import rows, split
from run_columbia_full_stress import descriptor

CFG=REPO/'configs/columbia_full_detection.json'; EC=REPO/'configs/evidence_replay.json'
def main():
 cfg=json.loads(CFG.read_text()); ec=json.loads(EC.read_text()); root=REPO/cfg['dataset_root']; rs=rows(root); sp=split(rs,cfg['split_ratios'],cfg['seed'])
 x=lambda q:np.stack([descriptor(Image.open(r['path'])) for r in q]); y=lambda q:np.array([r['_y'] for r in q])
 model=fit(x(sp['train']),y(sp['train']),cfg); threshold=threshold_dev(y(sp['dev']),predict(model,x(sp['dev']))); scores=predict(model,x(sp['test']))
 out=REPO/'results/columbia_evidence'; out.mkdir(parents=True,exist_ok=True)
 artifact={'artifact_version':'forenx-logistic-0.1','feature_operation':'forenx.grayscale_stats.v1','mean':model['mu'].tolist(),'scale':model['sd'].tolist(),'weights':model['w'].tolist(),'threshold':threshold,'training_config_sha256':file_hash(CFG)}
 model_path=out/'model.json'; model_path.write_text(json.dumps(artifact,separators=(',',':'))+'\n')
 model_hash=file_hash(model_path); packages=[]
 for r,s in zip(sp['test'],scores):
  pred='spliced' if s>=threshold else 'authentic'; digest=file_hash(r['path']); eid=hashlib.sha256((digest+model_hash).encode()).hexdigest()
  packages.append({'schema_version':ec['schema_version'],'evidence_id':eid,'sample_id':r['sample_id'],
   'input':{'relative_path':r['relative_path'],'sha256':digest},
   'claim':{'type':'patch-statistics-compatible-classification','text':f'The frozen pilot classifier labels this image as {pred}.','scope':'Columbia Uncompressed benchmark domain only; not proof of authenticity or forgery.'},
   'observation':{'score_spliced':float(s),'threshold':threshold,'predicted_class':pred},
   'operation':{'id':'forenx.grayscale_stats.v1','description':'Convert to grayscale; compute fixed histogram, spatial mean and gradient descriptor; apply frozen logistic model.'},
   'model':{'artifact':'model.json','sha256':model_hash},
   'applicability':['Image decodable by Pillow','Benchmark-like full image','No claim beyond tested domain'],
   'alternatives':['Score may reflect camera, acquisition or resizing differences rather than manipulation'],
   'failure_modes':['Distribution shift','Resizing-induced score drift','Incomplete source-scene isolation'],
   'uncertainty':{'calibration_domain':'development split only','score_is_probability':False}})
 with (out/'evidence.jsonl').open('w',encoding='utf-8',newline='\n') as f:
  for p in packages:f.write(json.dumps(p,separators=(',',':'))+'\n')
 run={'status':'complete','decisions':len(sp['test']),'evidence_items':len(packages),'ECR':len(packages)/len(sp['test']),'model_sha256':model_hash,'created_utc':datetime.now(timezone.utc).isoformat()}
 (out/'generation.json').write_text(json.dumps(run,indent=2)+'\n'); print(json.dumps(run,indent=2))
if __name__=='__main__':main()
