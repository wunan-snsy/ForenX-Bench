#!/usr/bin/env python3
"""Paired bootstrap comparison of frozen TruFor and MVSS-Net on CASIA v1."""
import csv, json
from pathlib import Path
import numpy as np
from PIL import Image
from run_columbia_blocks_baseline import REPO, auc

RNG = np.random.default_rng(20261004)

def paths(model, row):
    image = Path(row['relative_path'])
    if model == 'trufor':
        folder = 'trufor_casia_v1_authentic' if row['class'] == 'authentic' else 'trufor_casia_v1_tampered'
        rel = Path(image.name) if row['class'] == 'authentic' else Path(*image.parts[1:])
        return REPO/'results'/folder/(str(rel)+'.npz')
    folder = 'mvss_casia_v1_authentic' if row['class'] == 'authentic' else 'mvss_casia_v1_tampered'
    return REPO/'results'/folder/(image.name+'.npz')

def f1(y, s):
    p=s>=.5; tp=((y==1)&p).sum(); fp=((y==0)&p).sum(); fn=((y==1)&~p).sum()
    return float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0.

def local_f1(a, m):
    p=a>=.5; tp=(p&m).sum(); fp=(p&~m).sum(); fn=(~p&m).sum()
    return float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0.

def paired_delta(metric, n, reps=5000):
    values=[]
    for _ in range(reps):
        ix=RNG.integers(0,n,n); values.append(metric(ix))
    values=np.asarray(values)
    tail=min(((values<=0).sum()+1)/(len(values)+1),((values>=0).sum()+1)/(len(values)+1))
    return {'point': float(metric(np.arange(n))), 'bootstrap_95ci': [float(np.quantile(values,.025)),float(np.quantile(values,.975))],
            'p_two_sided_sign': float(min(1.,2*tail))}

def main():
    with (REPO/'manifests/casia_v1_manifest.csv').open(encoding='utf-8') as h: rows=list(csv.DictReader(h))
    y=np.array([r['class']=='tampered' for r in rows],dtype=int)
    tru=np.array([float(np.load(paths('trufor',r))['score']) for r in rows]); mv=np.array([float(np.load(paths('mvss',r))['score']) for r in rows])
    masked=[r for r in rows if r['mask_path']]; root=REPO/'data/external/casia_v1_corrected'
    tf=[]; mf=[]
    for r in masked:
        mask=np.asarray(Image.open(root/r['mask_path']).convert('L'))>127
        tf.append(local_f1(np.asarray(np.load(paths('trufor',r))['map']),mask))
        mf.append(local_f1(np.asarray(np.load(paths('mvss',r))['map']),mask))
    tf=np.asarray(tf); mf=np.asarray(mf)
    results={'dataset':'CASIA v1 corrected + Pham ground truth','unit':'paired images; 5,000 nonparametric bootstrap resamples','threshold':.5,
      'trufor_minus_mvss':{
        'detection_auc':paired_delta(lambda ix:auc(y[ix],tru[ix])-auc(y[ix],mv[ix]),len(rows)),
        'detection_f1':paired_delta(lambda ix:f1(y[ix],tru[ix])-f1(y[ix],mv[ix]),len(rows)),
        'localization_f1_macro':paired_delta(lambda ix:tf[ix].mean()-mf[ix].mean(),len(masked))}}
    out=REPO/'results/casia_v1_statistics';out.mkdir(parents=True,exist_ok=True)
    (out/'paired_bootstrap.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
