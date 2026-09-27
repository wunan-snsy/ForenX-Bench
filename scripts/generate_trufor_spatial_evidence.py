#!/usr/bin/env python3
import csv,hashlib,json
from pathlib import Path
import numpy as np
from run_columbia_blocks_baseline import REPO,file_hash

def array_hash(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def main():
 cfg=json.loads((REPO/'configs/trufor_evidence_replay.json').read_text()); root=REPO/'data/external/columbia_uncompressed/4cam_splc'; outroot=REPO/'results/trufor_columbia_spliced'
 metrics={r['sample_id']:r for r in csv.DictReader((REPO/'results/trufor_columbia_boundary/per_image.csv').open())}; items=[]
 for p in sorted(root.glob('*.tif')):
  op=outroot/(p.name+'.npz'); z=np.load(op); m=metrics[p.stem]
  items.append({'schema_version':cfg['schema_version'],'evidence_id':hashlib.sha256((file_hash(p)+file_hash(op)).encode()).hexdigest(),'sample_id':p.stem,
   'input':{'relative_path':p.relative_to(REPO).as_posix(),'sha256':file_hash(p)},'claim':{'text':'TruFor reports a spatial anomaly map and image-level manipulation score for this benchmark image.','scope':'Model observation only; not unique proof of forgery.'},
   'operation':{'id':'grip-unina.trufor.ae54475','repository_commit':'ae54475df6f41a491d7615100feb19263dec13f7','weights_md5':'7bee48f3476c75616c3c5721ab256ff8'},
   'observation':{'image_score':float(z['score']),'map_shape':list(z['map'].shape),'map_sha256':array_hash(z['map']),'confidence_sha256':array_hash(z['conf']),'map_mean':float(z['map'].mean()),'confidence_mean':float(z['conf'].mean())},
   'reference':{'definition':cfg['reference_definition'],'scope':cfg['reference_scope'],'fixed_threshold':0.5,'precision':float(m['precision']),'recall':float(m['recall']),'f1':float(m['f1']),'iou':float(m['iou']),'mcc':float(m['mcc'])},
   'applicability':['Nonprofit research use','Image decodable by Pillow','TruFor tested execution environment'],'alternatives':['Camera pipeline anomaly','Resampling or compression','Reference and anomaly-map semantics differ'],'failure_modes':['Distribution shift','Environment-dependent floating point variation','Over-broad anomaly region'],'uncertainty':{'artifact':'confidence map','statement':'Confidence is model-derived reliability, not empirical correctness probability.'}})
 out=REPO/'results/trufor_spatial_evidence';out.mkdir(parents=True,exist_ok=True)
 with (out/'evidence.jsonl').open('w',encoding='utf-8') as f:
  for x in items:f.write(json.dumps(x,separators=(',',':'))+'\n')
 summary={'status':'complete','decisions':180,'evidence_items':len(items),'ECR':len(items)/180,'ETR':sum(all(k in x for k in ['input','claim','operation','observation','reference','applicability','alternatives','failure_modes','uncertainty']) for x in items)/len(items)}
 (out/'generation.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
