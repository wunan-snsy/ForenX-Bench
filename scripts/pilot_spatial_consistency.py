#!/usr/bin/env python3
"""Test whether transform-map consistency identifies poor clean localization."""
import csv, hashlib, json, sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).parent))
from evaluate_casia_v1_models import prediction_path, localize, source_group
from pilot_stability_calibration import auc, fit_logistic, predict
from run_columbia_blocks_baseline import REPO

def main():
    with (REPO/'manifests/casia_v1_manifest.csv').open(encoding='utf-8') as h: rows=[r for r in csv.DictReader(h) if r['mask_path']]
    stress=json.loads((REPO/'results/casia_v1_full_stress_inputs/manifest.json').read_text()); by={}
    for x in stress: by.setdefault(x['sample_id'],[]).append(x)
    root=REPO/'data/external/casia_v1_corrected'; out={'dataset':'CASIA v1','target':'clean localization IoU below 0.50','models':{}}
    for model in ('trufor','mvss'):
        records=[]
        for r in rows:
            z=np.load(prediction_path(model,r)); base=np.asarray(z['map'],dtype=np.float32); ref=np.asarray(Image.open(root/r['mask_path']).convert('L'))>127; iou=localize(base,ref,.5)['iou']; diff=[]; agree=[]
            for x in by[r['sample_id']]:
                p=REPO/'results'/f'{model}_casia_v1_full_stress_outputs'/x['condition']/x['class']/(Path(x['path']).name+'.npz'); m=np.asarray(np.load(p)['map'],dtype=np.float32); diff.append(np.abs(m-base).mean()); agree.append(((m>=.5)==(base>=.5)).mean())
            g=source_group(r['sample_id'],r['class']); records.append((g,int(iou<.5),[float(np.mean(diff)),float(np.min(agree)),float(np.mean(agree))]))
        tr=[x for x in records if int(hashlib.sha256(x[0].encode()).hexdigest()[:8],16)%100<60]; te=[x for x in records if int(hashlib.sha256(x[0].encode()).hexdigest()[:8],16)%100>=60]
        fit=fit_logistic(np.array([x[2] for x in tr]),[x[1] for x in tr]); risk=predict(fit,np.array([x[2] for x in te]))
        out['models'][model]={'n_train':len(tr),'n_holdout':len(te),'poor_localization_rate':float(np.mean([x[1] for x in te])),'poor_localization_risk_auc':auc([x[1] for x in te],risk)}
    target=REPO/'results/spatial_consistency_pilot'; target.mkdir(parents=True,exist_ok=True); (target/'summary.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))

if __name__=='__main__': main()
