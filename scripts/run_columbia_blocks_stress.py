#!/usr/bin/env python3
"""Controlled digital post-processing stress test on held-out Columbia image blocks."""
from __future__ import annotations
import csv, hashlib, io, json
from pathlib import Path
import numpy as np
from PIL import Image
from run_columbia_blocks_baseline import (REPO, auc, file_hash, fit, load_rows,
    predict, score_metrics, split_groups, threshold_dev, auc_ci)

def image_features(image):
    x=np.asarray(image.convert('L'),dtype=np.float64)/255
    hist,_=np.histogram(x,bins=32,range=(0,1)); z=(hist/hist.sum()).tolist()
    z += [float(x.mean()),float(x.std()),*[float(np.percentile(x,p)) for p in (10,50,90)]]
    for y in np.array_split(x,4,axis=0):
        for cell in np.array_split(y,4,axis=1): z.append(float(cell.mean()))
    for g in (np.diff(x,axis=1),np.diff(x,axis=0)):
        z += [float(np.abs(g).mean()),float(g.std()),float(np.percentile(np.abs(g),90))]
        h,_=np.histogram(g,bins=16,range=(-1,1)); z += (h/h.sum()).tolist()
    return np.asarray(z)

def transform_jpeg(path,quality):
    im=Image.open(path).convert('L'); b=io.BytesIO(); im.save(b,format='JPEG',quality=quality,optimize=False,progressive=False)
    b.seek(0)
    with Image.open(b) as out: return out.convert('L').copy()

def transform_resize(path,scale):
    im=Image.open(path).convert('L'); n=max(1,round(128*scale))
    small=im.resize((n,n),Image.Resampling.LANCZOS)
    return small.resize((128,128),Image.Resampling.BICUBIC)

def main():
    cfg=json.loads((REPO/'configs/columbia_blocks_baseline.json').read_text(encoding='utf-8'))
    stress=json.loads((REPO/'configs/columbia_blocks_stress.json').read_text(encoding='utf-8'))
    rows=load_rows(REPO/cfg['manifest'],REPO/cfg['dataset_root'])
    split,_=split_groups(rows,cfg['split_group_ratios'],int(cfg['seed']))
    cachekey=hashlib.sha256((file_hash(REPO/cfg['manifest'])+cfg['features']).encode()).hexdigest()
    cachepath=REPO/'data/external/columbia_image_splicing_blocks/features.npz'
    z=np.load(cachepath,allow_pickle=False)
    if str(z['cache_key'])!=cachekey: raise ValueError('Feature cache does not match frozen manifest/config')
    cache={str(i):v for i,v in zip(z['sample_ids'],z['features'])}
    x=lambda rs:np.stack([cache[r['sample_id']] for r in rs]); y=lambda rs:np.array([r['_y'] for r in rs])
    model=fit(x(split['train']),y(split['train']),cfg)
    devscore=predict(model,x(split['dev'])); threshold=threshold_dev(y(split['dev']),devscore)
    test=split['test']; base_score=predict(model,x(test)); base_metrics=score_metrics(y(test),base_score,threshold)
    conditions={'original':[(r['_path'],lambda p:Image.open(p).convert('L')) for r in test]}
    for q in stress['jpeg_quality']:
        conditions[f'jpeg_q{q}']=[(r['_path'],lambda p,q=q:transform_jpeg(p,q)) for r in test]
    for scale in stress['resize_scale']:
        conditions[f'resize_{scale}']=[(r['_path'],lambda p,s=scale:transform_resize(p,s)) for r in test]
    results={}; all_scores={'original':base_score}
    for name,items in conditions.items():
        scores=base_score if name=='original' else np.array([predict(model,image_features(fn(path))[None,:])[0] for path,fn in items])
        all_scores[name]=scores; m=score_metrics(y(test),scores,threshold)
        m['prediction_flip_rate_vs_original']=float(np.mean((scores>=threshold)!=(base_score>=threshold)))
        m['mean_absolute_score_change_vs_original']=float(np.mean(np.abs(scores-base_score)))
        m['roc_auc_change_vs_original']=float(m['roc_auc']-base_metrics['roc_auc'])
        m['roc_auc_group_bootstrap_95ci']=auc_ci(test,scores,int(cfg['bootstrap_replicates']),int(cfg['bootstrap_seed']))
        results[name]=m
    out=REPO/'results/columbia_blocks_stress'; out.mkdir(parents=True,exist_ok=True)
    (out/'stress.json').write_text(json.dumps({'status':'complete','dataset_task':'patch-level classification','warning':stress['note'],
      'config_sha256':file_hash(REPO/'configs/columbia_blocks_stress.json'),'baseline_config_sha256':file_hash(REPO/'configs/columbia_blocks_baseline.json'),
      'test_manifest_sha256':file_hash(REPO/cfg['manifest']),'n_test':len(test),'threshold_frozen_from_dev':threshold,
      'transformations':{'jpeg_quality':stress['jpeg_quality'],'resize_scale':stress['resize_scale'],'downsample':stress['resize_downsample'],'restore':stress['resize_restore']},
      'results':results},indent=2)+'\n',encoding='utf-8')
    with (out/'stress_predictions.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['sample_id','condition','score_spliced','predicted_class'])
        for name,scores in all_scores.items():
            for r,s in zip(test,scores): w.writerow([r['sample_id'],name,f'{s:.10f}','spliced' if s>=threshold else 'authentic'])
    print(json.dumps({'status':'complete','n_test':len(test),'threshold':threshold,'results':results,'outputs':str(out)},indent=2))

if __name__=='__main__': main()
