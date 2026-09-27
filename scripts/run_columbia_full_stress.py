#!/usr/bin/env python3
import io, json, hashlib
import numpy as np
from PIL import Image
from run_columbia_blocks_baseline import REPO, fit, predict, threshold_dev, score_metrics, auc_ci
from run_columbia_full_detection import rows, split

CFG=REPO/'configs/columbia_full_detection.json'; STRESS=REPO/'configs/columbia_full_stress.json'

def descriptor(im):
 x=np.asarray(im.convert('L'),dtype=np.float64)/255
 hist,_=np.histogram(x,bins=32,range=(0,1)); z=(hist/hist.sum()).tolist()
 z += [float(x.mean()),float(x.std()),*[float(np.percentile(x,p)) for p in (10,50,90)]]
 for band in np.array_split(x,4,axis=0):
  for cell in np.array_split(band,4,axis=1): z.append(float(cell.mean()))
 for g in (np.diff(x,axis=1),np.diff(x,axis=0)):
  z += [float(np.abs(g).mean()),float(g.std()),float(np.percentile(np.abs(g),90))]
  h,_=np.histogram(g,bins=16,range=(-1,1)); z += (h/h.sum()).tolist()
 return np.asarray(z)

def transform(path,kind,value):
 im=Image.open(path).convert('L')
 if kind=='jpeg':
  b=io.BytesIO(); im.save(b,'JPEG',quality=value,progressive=False,optimize=False); b.seek(0)
  with Image.open(b) as q:return q.convert('L').copy()
 n=(max(1,round(im.width*value)),max(1,round(im.height*value)))
 return im.resize(n,Image.Resampling.LANCZOS).resize(im.size,Image.Resampling.BICUBIC)

def main():
 cfg=json.loads(CFG.read_text()); sc=json.loads(STRESS.read_text()); rs=rows(REPO/cfg['dataset_root']); sp=split(rs,cfg['split_ratios'],cfg['seed'])
 feats=lambda q:np.stack([descriptor(Image.open(r['path'])) for r in q]); yy=lambda q:np.array([r['_y'] for r in q])
 model=fit(feats(sp['train']),yy(sp['train']),cfg); threshold=threshold_dev(yy(sp['dev']),predict(model,feats(sp['dev'])))
 test=sp['test']; base=predict(model,feats(test)); conditions=[('original',None,None)]+[(f'jpeg_q{q}','jpeg',q) for q in sc['jpeg_quality']]+[(f'resize_{s}','resize',s) for s in sc['resize_scale']]
 result={}
 for name,kind,val in conditions:
  scores=base if kind is None else predict(model,np.stack([descriptor(transform(r['path'],kind,val)) for r in test]))
  m=score_metrics(yy(test),scores,threshold); m['roc_auc_group_bootstrap_95ci']=auc_ci(test,scores,cfg['bootstrap_replicates'],cfg['bootstrap_seed'])
  m['delta_auc']=m['roc_auc']-score_metrics(yy(test),base,threshold)['roc_auc']; m['flip_rate']=float(np.mean((scores>=threshold)!=(base>=threshold))); m['mean_abs_score_change']=float(np.mean(abs(scores-base)))
  result[name]=m
 out=REPO/'results/columbia_full_stress'; out.mkdir(parents=True,exist_ok=True)
 record={'status':'complete','warning':sc['note'],'n_test':len(test),'threshold_dev':threshold,'results':result,'config_sha256':hashlib.sha256(STRESS.read_bytes()).hexdigest()}
 (out/'stress.json').write_text(json.dumps(record,indent=2)+'\n'); print(json.dumps(record,indent=2))
if __name__=='__main__':main()
