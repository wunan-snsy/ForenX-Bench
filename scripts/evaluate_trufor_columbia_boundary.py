#!/usr/bin/env python3
"""Evaluate frozen TruFor maps against Columbia blue splice-boundary references."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from run_columbia_blocks_baseline import REPO

CFG=REPO/'configs/columbia_edge_boundary.json'; THRESHOLD=.5
def decode_boundary(path):
 rgb=np.asarray(Image.open(path).convert('RGB'),dtype=np.float32); proto=np.array([[200,0,0],[0,200,0],[0,0,200]],dtype=np.float32)
 return ((rgb[...,None,:]-proto)**2).sum(-1).argmin(-1)==2
def main():
 root=REPO/'data/external/columbia_uncompressed/4cam_splc'; predroot=REPO/'results/trufor_columbia_spliced'; rows=[]; failures=[]
 for p in sorted(root.glob('*.tif')):
  q=predroot/(p.name+'.npz'); m=root/'edgemask'/(p.stem+'_edgemask_3.jpg')
  try:
   z=np.load(q); score=np.asarray(z['map'],dtype=np.float32); ref=decode_boundary(m)
   if score.shape!=ref.shape: raise ValueError(f'shape {score.shape}!={ref.shape}')
   pred=score>=THRESHOLD; tp=int((pred&ref).sum()); tn=int((~pred&~ref).sum()); fp=int((pred&~ref).sum()); fn=int((~pred&ref).sum())
   precision=tp/(tp+fp) if tp+fp else 0.; recall=tp/(tp+fn) if tp+fn else 0.; f1=2*precision*recall/(precision+recall) if precision+recall else 0.; iou=tp/(tp+fp+fn) if tp+fp+fn else 0.
   den=((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**.5; mcc=(tp*tn-fp*fn)/den if den else 0.
   rows.append({'sample_id':p.stem,'image_score':float(z['score']),'precision':precision,'recall':recall,'f1':f1,'iou':iou,'mcc':mcc,'reference_prevalence':float(ref.mean()),'prediction_prevalence':float(pred.mean()),'mean_map_reference':float(score[ref].mean()),'mean_map_elsewhere':float(score[~ref].mean())})
  except Exception as e:failures.append({'sample_id':p.stem,'error':str(e)})
 keys=['precision','recall','f1','iou','mcc','reference_prevalence','prediction_prevalence','mean_map_reference','mean_map_elsewhere','image_score']
 summary={'status':'complete','n_expected':180,'n_evaluated':len(rows),'threshold':THRESHOLD,'reference':'nearest-prototype decoded blue region in _edgemask_3','macro':{k:float(np.mean([r[k] for r in rows])) for k in keys},'median':{k:float(np.median([r[k] for r in rows])) for k in keys},'failures':failures,'trufor_weights_md5':'7bee48f3476c75616c3c5721ab256ff8'}
 out=REPO/'results/trufor_columbia_boundary'; out.mkdir(parents=True,exist_ok=True); (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 with (out/'per_image.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
