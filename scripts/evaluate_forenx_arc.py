#!/usr/bin/env python3
"""ForenX-ARC: transform-consistency error-risk calibration evaluation.

Uses only frozen detector outputs. Risk features are clean score margin and
predeclared stress-score stability; a logistic risk head is fit on CASIA train
groups and evaluated on held-out CASIA and, when available, COVERAGE.
"""
import csv, hashlib, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from evaluate_casia_v1_models import prediction_path as casia_path, source_group
from run_columbia_blocks_baseline import REPO
from pilot_stability_calibration import fit_logistic, predict, auc, brier

CONDS=('jpeg_q80','jpeg_q40','resize_0.5','resize_0.25')
def coverage_path(model,row): return REPO/'results'/f'{model}_coverage_{row["class"]}'/(Path(row['relative_path']).name+'.npz')
def collect(dataset,model):
    manifest=REPO/'manifests'/('casia_v1_manifest.csv' if dataset=='casia' else 'coverage_manifest.csv')
    input_name = 'casia_v1_full_stress_inputs' if dataset == 'casia' else 'coverage_full_stress_inputs'
    stress=json.loads((REPO/'results'/input_name/'manifest.json').read_text())
    by={};[by.setdefault(x['sample_id'],[]).append(x) for x in stress]
    rows=[]
    with manifest.open(encoding='utf-8') as h:
      for r in csv.DictReader(h):
        base=float(np.load(casia_path(model,r) if dataset=='casia' else coverage_path(model,r))['score']); pred=int(base>=.5); label=int(r['class']=='tampered'); vals=[]
        output_name = f'{model}_casia_v1_full_stress_outputs' if dataset == 'casia' else f'{model}_coverage_full_stress_outputs'
        for x in by[r['sample_id']]: vals.append(float(np.load(REPO/'results'/output_name/x['condition']/x['class']/(Path(x['path']).name+'.npz'))['score']))
        d=np.abs(np.array(vals)-base); group=source_group(r['sample_id'],r['class']) if dataset=='casia' else 'pair:'+r['pair_id']
        rows.append({'group':group,'error':1-int(pred==label),'f':np.array([abs(base-.5),d.mean(),d.max(),np.mean((np.array(vals)>=.5)!=pred)])})
    return rows
def metrics(rows,risk):
    y=np.array([r['error'] for r in rows]);out={'error_risk_auc':auc(y,risk),'error_risk_brier':brier(y,risk),'baseline_error_rate':float(y.mean())}
    for c in (.6,.7,.8,.9):
      kept=np.argsort(risk)[:round(len(y)*c)];out[f'risk_at_{int(c*100)}pct_coverage']=float(y[kept].mean())
    return out
def main():
  result={'method':'ForenX-ARC v0.1','features':['clean_score_margin','mean_abs_stress_score_delta','max_abs_stress_score_delta','stress_threshold_flip_fraction'],'models':{}}
  for m in ('trufor','mvss'):
    casia=collect('casia',m); train=[r for r in casia if int(hashlib.sha256(r['group'].encode()).hexdigest()[:8],16)%100<60]; test=[r for r in casia if int(hashlib.sha256(r['group'].encode()).hexdigest()[:8],16)%100>=60]
    fit=fit_logistic(np.array([r['f'] for r in train]),[r['error'] for r in train]); arc=predict(fit,np.array([r['f'] for r in test])); confidence=np.array([1-2*r['f'][0] for r in test])
    item={'casia_holdout':{'n_train':len(train),'n_test':len(test),'forenx_arc':metrics(test,arc),'confidence_only':metrics(test,confidence)}}
    try:
      cov=collect('coverage',m); item['coverage_external']={'n':len(cov),'forenx_arc':metrics(cov,predict(fit,np.array([r['f'] for r in cov]))),'confidence_only':metrics(cov,np.array([1-2*r['f'][0] for r in cov]))}
    except FileNotFoundError: item['coverage_external']={'status':'pending'}
    result['models'][m]=item
  out=REPO/'results/forenx_arc';out.mkdir(parents=True,exist_ok=True);(out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
