#!/usr/bin/env python3
"""Train spatial-consistency risk on CASIA and test it on COVERAGE."""
import csv,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).parent))
from evaluate_casia_v1_models import prediction_path,localize
from evaluate_forenx_arc import coverage_path
from pilot_stability_calibration import auc,fit_logistic,predict
from run_columbia_blocks_baseline import REPO
def rows(ds,model):
 name='casia_v1' if ds=='casia' else 'coverage'; man=REPO/'manifests'/('casia_v1_manifest.csv' if ds=='casia' else 'coverage_manifest.csv'); root=REPO/('data/external/casia_v1_corrected' if ds=='casia' else 'data/external/COVERAGE'); stress=json.loads((REPO/'results'/f'{name}_full_stress_inputs'/'manifest.json').read_text()); by={}
 for x in stress:by.setdefault(x['sample_id'],[]).append(x)
 out=[]
 with man.open(encoding='utf-8') as h:
  for r in csv.DictReader(h):
   if not r['mask_path']:continue
   z=np.load(prediction_path(model,r) if ds=='casia' else coverage_path(model,r)); b=np.asarray(z['map'],dtype=np.float32); score=float(z['score']); ref=np.asarray(Image.open(root/r['mask_path']).convert('L'))>127;iou=localize(b,ref,.5)['iou']; dsout=f'{model}_{name}_full_stress_outputs';d=[];a=[]
   for x in by[r['sample_id']]:
    m=np.asarray(np.load(REPO/'results'/dsout/x['condition']/x['class']/(Path(x['path']).name+'.npz'))['map'],dtype=np.float32);d.append(np.abs(m-b).mean());a.append(((m>=.5)==(b>=.5)).mean())
   out.append((int(iou<.5),[np.mean(d),np.min(a),np.mean(a)],1-2*abs(score-.5)))
 return out
def main():
 out={}
 for m in ('trufor','mvss'):
  tr=rows('casia',m);te=rows('coverage',m);fit=fit_logistic(np.array([x[1] for x in tr]),[x[0] for x in tr]);risk=predict(fit,np.array([x[1] for x in te]));y=np.array([x[0] for x in te]);q={'n_train':len(tr),'n_external':len(te),'poor_rate':float(y.mean()),'risk_auc':auc(y,risk),'confidence_only_auc':auc(y,[x[2] for x in te])}
  for c in (.6,.7,.8,.9):
   n=round(len(y)*c);q[f'arc_poor_rate_at_{int(c*100)}pct_coverage']=float(y[np.argsort(risk)[:n]].mean());q[f'confidence_poor_rate_at_{int(c*100)}pct_coverage']=float(y[np.argsort([x[2] for x in te])[:n]].mean())
  out[m]=q
 target=REPO/'results/spatial_external_pilot';target.mkdir(parents=True,exist_ok=True);(target/'summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
