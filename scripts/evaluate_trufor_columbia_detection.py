#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
from run_columbia_blocks_baseline import REPO, score_metrics, auc_ci
from run_columbia_full_detection import rows

def main():
 root=REPO/'data/external/columbia_uncompressed'; rs=rows(root); scores=[]; failures=[]
 for r in rs:
  folder='trufor_columbia_authentic' if r['class']=='authentic' else 'trufor_columbia_spliced'; p=REPO/'results'/folder/(r['path'].name+'.npz')
  try:scores.append(float(np.load(p)['score']))
  except Exception as e:failures.append({'sample_id':r['sample_id'],'error':str(e)}); scores.append(float('nan'))
 ok=np.isfinite(scores); used=[r for r,k in zip(rs,ok) if k]; s=np.asarray(scores)[ok]; y=np.array([r['_y'] for r in used]); m=score_metrics(y,s,.5); m['roc_auc_group_bootstrap_95ci']=auc_ci(used,s,2000,20260928)
 result={'status':'complete','threshold':.5,'n_expected':363,'n_evaluated':len(used),'metrics':m,'score_by_class':{'authentic_mean':float(s[y==0].mean()),'authentic_median':float(np.median(s[y==0])),'spliced_mean':float(s[y==1].mean()),'spliced_median':float(np.median(s[y==1]))},'failures':failures,'weights_md5':'7bee48f3476c75616c3c5721ab256ff8'}
 out=REPO/'results/trufor_columbia_detection';out.mkdir(parents=True,exist_ok=True);(out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
