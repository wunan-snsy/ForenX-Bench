#!/usr/bin/env python3
"""Reproducible patch-level baseline for the restricted Columbia block dataset."""
from __future__ import annotations
import csv, hashlib, json, math, platform, random, sys, time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
CFG_PATH = REPO / "configs/columbia_blocks_baseline.json"

def file_hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''): h.update(chunk)
    return h.hexdigest()

def build_manifest(root, manifest):
    files=sorted(root.rglob('*.bmp'))
    if not files: raise FileNotFoundError(f'No BMP files under {root}; acquire and extract the dataset first')
    rows=[]
    for p in files:
        rel=p.relative_to(root).as_posix(); folder=p.parent.name.lower()
        label='authentic' if folder.startswith('au-') else 'spliced' if folder.startswith('sp-') else None
        if label is None: raise ValueError(f'Unrecognized class directory: {p.parent.name}')
        with Image.open(p) as im:
            if im.size!=(128,128): raise ValueError(f'Unexpected dimensions {im.size}: {rel}')
            if im.mode not in ('L','P'): raise ValueError(f'Unexpected BMP mode {im.mode}: {rel}')
        rows.append({'sample_id':p.stem,'relative_path':rel,'class':label,'subclass':p.parent.name,
                     'sha256':file_hash(p),'bytes':p.stat().st_size,'width':128,'height':128,'bit_depth':8})
    groups=defaultdict(list)
    for r in rows: groups[r['sha256']].append(r)
    for digest,members in groups.items():
        group_id='sha256:'+digest if len(members)>1 else members[0]['sample_id']
        for r in members: r['source_group']=group_id
    manifest.parent.mkdir(parents=True,exist_ok=True)
    with manifest.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['sample_id','relative_path','class','subclass','source_group','sha256','bytes','width','height','bit_depth'])
        w.writeheader(); w.writerows(rows)
    return len(rows)

def load_rows(manifest,root):
    if not manifest.exists():
        n=build_manifest(root,manifest)
        print(f'Built local manifest for {n} files at {manifest}')
    with open(manifest,newline='',encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
    for r in rows:
        p=root/r['relative_path']
        if not p.is_file(): raise FileNotFoundError(p)
        if file_hash(p)!=r['sha256']: raise ValueError('SHA-256 mismatch: '+r['relative_path'])
        if r['class'] not in ('authentic','spliced'): raise ValueError('Unexpected label')
        r['_path']=p; r['_y']=int(r['class']=='spliced')
    if not rows: raise ValueError('Empty manifest')
    return rows

def split_groups(rows,ratios,seed):
    if not math.isclose(sum(ratios.values()),1): raise ValueError('Split ratios must sum to 1')
    grouped=defaultdict(list)
    for r in rows: grouped[r['source_group']].append(r)
    labels=defaultdict(list)
    for gid,items in grouped.items():
        ys={r['_y'] for r in items}
        if len(ys)!=1: raise ValueError('Duplicate group crosses labels: '+gid)
        labels[next(iter(ys))].append((gid,items))
    names=list(ratios); rng=random.Random(seed)
    out={n:[] for n in names}; gids={n:[] for n in names}
    for y,groups in sorted(labels.items()):
        rng.shuffle(groups); groups.sort(key=lambda q:-len(q[1]))
        target={n:ratios[n]*sum(len(v) for _,v in groups) for n in names}
        count={n:0 for n in names}
        for gid,items in groups:
            n=max(names,key=lambda k:(target[k]-count[k],ratios[k]))
            out[n].extend(items); gids[n].append(gid); count[n]+=len(items)
    return out,gids

def features(path):
    x=np.asarray(Image.open(path).convert('L'),dtype=np.float64)/255
    hist,_=np.histogram(x,bins=32,range=(0,1)); z=(hist/hist.sum()).tolist()
    z += [float(x.mean()),float(x.std()),*[float(np.percentile(x,p)) for p in (10,50,90)]]
    for y in np.array_split(x,4,axis=0):
        for cell in np.array_split(y,4,axis=1): z.append(float(cell.mean()))
    for g in (np.diff(x,axis=1),np.diff(x,axis=0)):
        z += [float(np.abs(g).mean()),float(g.std()),float(np.percentile(np.abs(g),90))]
        h,_=np.histogram(g,bins=16,range=(-1,1)); z += (h/h.sum()).tolist()
    return np.asarray(z)

def sigmoid(z): return 1/(1+np.exp(-np.clip(z,-40,40)))

def fit(x,y,cfg):
    mu=x.mean(0); sd=x.std(0); sd[sd<1e-12]=1
    a=np.column_stack([np.ones(len(x)),(x-mu)/sd]); w=np.zeros(a.shape[1]); last=float('inf')
    reg=float(cfg['regularization']); lr=float(cfg['learning_rate'])
    for it in range(int(cfg['max_iterations'])):
        p=sigmoid(a@w); loss=-np.mean(y*np.log(p+1e-12)+(1-y)*np.log(1-p+1e-12))+.5*reg*np.sum(w[1:]**2)
        grad=a.T@(p-y)/len(y); grad[1:]+=reg*w[1:]; w-=lr*grad
        if abs(last-loss)<float(cfg['tolerance']): break
        last=loss
    return {'mu':mu,'sd':sd,'w':w,'iterations':it+1,'loss':float(loss)}

def predict(m,x): return sigmoid(np.column_stack([np.ones(len(x)),(x-m['mu'])/m['sd']])@m['w'])

def auc(y,s):
    order=np.argsort(s,kind='mergesort'); ranks=np.empty(len(y)); i=0
    while i<len(order):
        j=i+1
        while j<len(order) and s[order[j]]==s[order[i]]: j+=1
        ranks[order[i:j]]=(i+1+j)/2; i=j
    n1=int(y.sum()); n0=len(y)-n1
    return float((ranks[y==1].sum()-n1*(n1+1)/2)/(n1*n0)) if n0 and n1 else float('nan')

def score_metrics(y,s,t):
    p=s>=t; tp=int(((y==1)&p).sum()); tn=int(((y==0)&~p).sum()); fp=int(((y==0)&p).sum()); fn=int(((y==1)&~p).sum())
    prec=tp/(tp+fp) if tp+fp else 0; rec=tp/(tp+fn) if tp+fn else 0; ece=0.
    for lo in np.arange(0,1,.1):
        m=(s>=lo)&(s<(lo+.1) if lo<.9 else s<=1)
        if m.any(): ece+=m.mean()*abs(float(s[m].mean())-float(y[m].mean()))
    return {'n':len(y),'authentic':int((y==0).sum()),'spliced':int((y==1).sum()),'roc_auc':auc(y,s),'accuracy':(tp+tn)/len(y),
            'precision_spliced':prec,'recall_spliced':rec,'f1_spliced':2*prec*rec/(prec+rec) if prec+rec else 0,
            'confusion_matrix':{'tn':tn,'fp':fp,'fn':fn,'tp':tp},'brier':float(np.mean((s-y)**2)),'ece_10bin':float(ece)}

def threshold_dev(y,s):
    best=None
    for t in np.unique(np.r_[0.,s,1.]):
        p=s>=t; tp=((y==1)&p).sum(); fn=((y==1)&~p).sum(); tn=((y==0)&~p).sum(); fp=((y==0)&p).sum()
        key=(float(tp/(tp+fn)-fp/(fp+tn)),-abs(float(t)-.5))
        if best is None or key>best[0]: best=(key,float(t))
    return best[1]

def auc_ci(rows,s,reps,seed):
    ix=defaultdict(list)
    for i,r in enumerate(rows): ix[r['source_group']].append(i)
    groups=list(ix); rng=np.random.default_rng(seed); vals=[]
    for _ in range(reps):
        chosen=rng.choice(groups,size=len(groups),replace=True); ids=np.concatenate([ix[g] for g in chosen]); y=np.array([rows[i]['_y'] for i in ids])
        if y.min()!=y.max(): vals.append(auc(y,s[ids]))
    return [float(np.quantile(vals,.025)),float(np.quantile(vals,.975))] if vals else [None,None]

def main():
    start=time.time(); cfg=json.loads(CFG_PATH.read_text(encoding='utf-8'))
    root=REPO/cfg['dataset_root']; manifest=REPO/cfg['manifest']; rows=load_rows(manifest,root)
    splits,group_ids=split_groups(rows,cfg['split_group_ratios'],int(cfg['seed']))
    key=hashlib.sha256((file_hash(manifest)+cfg['features']).encode()).hexdigest()
    cache_path=REPO/'data/external/columbia_image_splicing_blocks/features.npz'; cache={}
    if cache_path.exists():
        try:
            z=np.load(cache_path,allow_pickle=False)
            if str(z['cache_key'])==key: cache={str(i):v for i,v in zip(z['sample_ids'],z['features'])}
        except Exception: cache={}
    if len(cache)!=len(rows):
        cache={r['sample_id']:features(r['_path']) for r in rows}
        np.savez_compressed(cache_path,cache_key=np.array(key),sample_ids=np.array(list(cache)),features=np.stack(list(cache.values())))
    index={r['sample_id']:r for r in rows}; x=lambda rs:np.stack([cache[r['sample_id']] for r in rs]); y=lambda rs:np.array([r['_y'] for r in rs])
    model=fit(x(splits['train']),y(splits['train']),cfg); dev_s=predict(model,x(splits['dev'])); thr=threshold_dev(y(splits['dev']),dev_s)
    test=splits['test']; test_s=predict(model,x(test)); m=score_metrics(y(test),test_s,thr); m['roc_auc_group_bootstrap_95ci']=auc_ci(test,test_s,int(cfg['bootstrap_replicates']),int(cfg['bootstrap_seed']))
    out=REPO/'results/columbia_blocks_baseline'; out.mkdir(parents=True,exist_ok=True)
    with (out/'predictions.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['sample_id','relative_path','source_group','class','score_spliced','predicted_class','split']); w.writeheader()
        for r,s in zip(test,test_s): w.writerow({'sample_id':r['sample_id'],'relative_path':r['relative_path'],'source_group':r['source_group'],'class':r['class'],'score_spliced':f'{s:.10f}','predicted_class':'spliced' if s>=thr else 'authentic','split':'test'})
    split_record={n:{'images':len(splits[n]),'groups':len(group_ids[n]),'authentic':sum(r['_y']==0 for r in splits[n]),'spliced':sum(r['_y']==1 for r in splits[n]),'group_ids':sorted(group_ids[n])} for n in splits}
    if sum(set(group_ids[a])&set(group_ids[b])!=set() for a,b in [('train','dev'),('train','test'),('dev','test')]): raise RuntimeError('Group leakage')
    run={'run_id':datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'),'status':'complete','created_utc':datetime.now(timezone.utc).isoformat(),'dataset':cfg['dataset'],'task':cfg['task'],'warning':cfg['warning'],
         'source_manifest_sha256':file_hash(manifest),'config_sha256':file_hash(CFG_PATH),'split':split_record,'model':{'name':'handcrafted grayscale descriptor + NumPy L2 logistic regression','feature_dimensions':len(next(iter(cache.values()))),'iterations':model['iterations'],'final_training_loss':model['loss'],'regularization':cfg['regularization']},
         'threshold':{'value':thr,'selected_on':'dev','rule':cfg['threshold_selection']},'test_metrics':m,'runtime':{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'pillow':Image.__version__,'elapsed_seconds':round(time.time()-start,3)},'reproducibility':{'seed':cfg['seed'],'bootstrap_seed':cfg['bootstrap_seed'],'cache_key':key}}
    (out/'run.json').write_text(json.dumps(run,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({'run_id':run['run_id'],'status':run['status'],'split':{n:{k:v for k,v in q.items() if k!='group_ids'} for n,q in split_record.items()},'threshold_dev':thr,'test_metrics':m,'outputs':str(out)},indent=2))

if __name__=='__main__': main()
