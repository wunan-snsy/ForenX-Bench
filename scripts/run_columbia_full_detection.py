#!/usr/bin/env python3
"""Full-image detection pilot for Columbia Uncompressed dataset."""
import csv, json, random, re, hashlib
from pathlib import Path
import numpy as np
from run_columbia_blocks_baseline import REPO, features, fit, predict, threshold_dev, score_metrics, auc_ci

CFG=REPO/'configs/columbia_full_detection.json'

def rows(root):
 out=[]
 for label,folder in [('authentic','4cam_auth'),('spliced','4cam_splc')]:
  for p in sorted((root/folder).glob('*.tif')):
   stem=p.stem.lower()
   if label=='authentic':
    m=re.match(r'(.+?)_(\d+)_sub_\d+$',stem); group='auth:'+('_'.join(m.groups()) if m else stem)
   else:
    # Only camera-pair and item number are public; retain each item as a group.
    group='splc:'+stem
   out.append({'sample_id':p.stem,'path':p,'relative_path':p.relative_to(root).as_posix(),'class':label,'_y':int(label=='spliced'),'source_group':group})
 return out

def split(rs,ratios,seed):
 by={0:[],1:[]}; groups={}
 for r in rs: groups.setdefault(r['source_group'],[]).append(r)
 for g,x in groups.items(): by[x[0]['_y']].append((g,x))
 rng=random.Random(seed); names=list(ratios); out={n:[] for n in names}
 for y,gs in by.items():
  rng.shuffle(gs); gs.sort(key=lambda z:-len(z[1])); target={n:ratios[n]*sum(len(x) for _,x in gs) for n in names}; count={n:0 for n in names}
  for g,x in gs:
   n=max(names,key=lambda k:target[k]-count[k]); out[n]+=x; count[n]+=len(x)
 return out

def main():
 cfg=json.loads(CFG.read_text()); root=REPO/cfg['dataset_root']; rs=rows(root); sp=split(rs,cfg['split_ratios'],cfg['seed'])
 x=lambda q:np.stack([features(r['path']) for r in q]); y=lambda q:np.array([r['_y'] for r in q])
 model=fit(x(sp['train']),y(sp['train']),cfg); ds=predict(model,x(sp['dev'])); threshold=threshold_dev(y(sp['dev']),ds)
 ts=predict(model,x(sp['test'])); met=score_metrics(y(sp['test']),ts,threshold); met['roc_auc_group_bootstrap_95ci']=auc_ci(sp['test'],ts,cfg['bootstrap_replicates'],cfg['bootstrap_seed'])
 out=REPO/'results/columbia_full_detection'; out.mkdir(parents=True,exist_ok=True)
 with (out/'predictions.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.writer(f); w.writerow(['sample_id','class','source_group','score_spliced','prediction'])
  for r,s in zip(sp['test'],ts): w.writerow([r['sample_id'],r['class'],r['source_group'],f'{s:.10f}','spliced' if s>=threshold else 'authentic'])
 record={'status':'complete','warning':cfg['warning'],'counts':{k:{'images':len(v),'groups':len(set(r['source_group'] for r in v)),'authentic':sum(r['_y']==0 for r in v),'spliced':sum(r['_y']==1 for r in v)} for k,v in sp.items()},'threshold_dev':threshold,'test_metrics':met,'config_sha256':hashlib.sha256(CFG.read_bytes()).hexdigest()}
 (out/'run.json').write_text(json.dumps(record,indent=2)+'\n')
 print(json.dumps(record,indent=2))
if __name__=='__main__': main()
