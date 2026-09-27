#!/usr/bin/env python3
"""Evaluate a deterministic gradient cue against Columbia blue boundary references."""
import json, hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from run_columbia_blocks_baseline import REPO

CFG=REPO/'configs/columbia_edge_boundary.json'
def decode(rgb,prototypes):
 p=np.asarray(prototypes,dtype=np.float32); x=rgb.astype(np.float32)
 d=((x[...,None,:]-p[None,None,:,:])**2).sum(-1)
 idx=d.argmin(-1); ordered=np.partition(d,1,axis=-1)
 return idx, np.sqrt(ordered[...,1])-np.sqrt(ordered[...,0])

def gradient(im):
 x=np.asarray(im.convert('L'),dtype=np.float32)/255.; gx=np.zeros_like(x); gy=np.zeros_like(x)
 gx[:,1:-1]=(x[:,2:]-x[:,:-2])*.5; gy[1:-1,:]=(x[2:,:]-x[:-2,:])*.5
 return np.hypot(gx,gy)

def main():
 cfg=json.loads(CFG.read_text()); root=REPO/'data/external/columbia_uncompressed/4cam_splc'; keys=['camera_1','camera_2','boundary']; proto=[cfg['mask_prototypes_rgb'][k] for k in keys]
 rows=[]
 for imgp in sorted(root.glob('*.tif')):
  maskp=root/'edgemask'/(imgp.stem+'_edgemask_3.jpg')
  with Image.open(imgp) as im, Image.open(maskp) as mm:
   rgb=np.asarray(mm.convert('RGB')); cls,margin=decode(rgb,proto); ref=cls==2; score=gradient(im)
  n=int(ref.sum()); flat=score.ravel(); k=max(1,n); cut=np.partition(flat,len(flat)-k)[len(flat)-k]; pred=score>=cut
  tp=int((pred&ref).sum()); fp=int((pred&~ref).sum()); fn=int((~pred&ref).sum()); precision=tp/(tp+fp); recall=tp/(tp+fn); f1=2*precision*recall/(precision+recall) if precision+recall else 0
  prevalence=float(ref.mean()); enrichment=precision/prevalence if prevalence else None
  rows.append({'sample_id':imgp.stem,'reference_pixels':n,'reference_prevalence':prevalence,'precision':precision,'recall':recall,'f1':f1,'precision_enrichment':enrichment,'decode_margin_p01':float(np.quantile(margin,.01)),'decode_margin_median':float(np.median(margin))})
 vals=lambda k:[r[k] for r in rows if r[k] is not None]
 summary={'status':'complete','warning':cfg['warning'],'n_images':len(rows),'mask_decode':cfg['mask_decode'],'predictor':cfg['predictor'],'prediction_rule':cfg['prediction_rule'],
  'macro':{k:float(np.mean(vals(k))) for k in ['reference_prevalence','precision','recall','f1','precision_enrichment','decode_margin_p01','decode_margin_median']},
  'median':{k:float(np.median(vals(k))) for k in ['precision','recall','f1','precision_enrichment']},'config_sha256':hashlib.sha256(CFG.read_bytes()).hexdigest()}
 out=REPO/'results/columbia_edge_boundary'; out.mkdir(parents=True,exist_ok=True); (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');
 import csv
 with (out/'per_image.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
