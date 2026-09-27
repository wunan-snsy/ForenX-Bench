#!/usr/bin/env python3
"""Test predeclared transform-score fusion against a frozen clean-score baseline."""
import csv, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from evaluate_casia_v1_models import prediction_path as casia_path
from evaluate_forenx_arc import coverage_path
from run_columbia_blocks_baseline import REPO, score_metrics

def main():
 out={}
 for ds in ('casia','coverage'):
  name='casia_v1' if ds=='casia' else 'coverage'; manifest=REPO/'manifests'/('casia_v1_manifest.csv' if ds=='casia' else 'coverage_manifest.csv')
  stress=json.loads((REPO/'results'/f'{name}_full_stress_inputs'/'manifest.json').read_text()); by={}
  for x in stress: by.setdefault(x['sample_id'],[]).append(x)
  with manifest.open(encoding='utf-8') as h: rows=list(csv.DictReader(h))
  out[ds]={}
  for model in ('trufor','mvss'):
   clean=[]; mean=[]; median=[]; labels=[]
   oname=f'{model}_{name}_full_stress_outputs'
   for r in rows:
    b=float(np.load(casia_path(model,r) if ds=='casia' else coverage_path(model,r))['score']); vals=[b]
    for x in by[r['sample_id']]: vals.append(float(np.load(REPO/'results'/oname/x['condition']/x['class']/(Path(x['path']).name+'.npz'))['score']))
    clean.append(b);mean.append(np.mean(vals));median.append(np.median(vals));labels.append(int(r['class']=='tampered'))
   out[ds][model]={k:score_metrics(np.array(labels),np.array(v),.5) for k,v in {'clean':clean,'mean_5view':mean,'median_5view':median}.items()}
 target=REPO/'results/transform_ensemble_pilot';target.mkdir(parents=True,exist_ok=True);(target/'summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
