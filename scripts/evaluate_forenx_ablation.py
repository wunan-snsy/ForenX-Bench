"""Evaluate an ablation on CASIA v2 test and zero-shot COVERAGE."""
import csv,json,argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset,DataLoader
from run_columbia_blocks_baseline import REPO,score_metrics
from train_forenx_ablation import AblationNet

class EvalData(Dataset):
 def __init__(self,manifest,root,split=None):
  self.root=root;self.rows=list(csv.DictReader(manifest.open(encoding='utf-8')))
  if split is not None:self.rows=[r for r in self.rows if r.get('split')==split]
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  r=self.rows[i]
  with Image.open(self.root/r['relative_path']) as im:x=np.asarray(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR),dtype=np.float32)/255
  m=np.zeros((256,256),dtype=np.float32);valid=bool(r.get('mask_path',''))
  if valid:
   with Image.open(self.root/r['mask_path']) as im:m=(np.asarray(im.convert('L').resize((256,256),Image.Resampling.NEAREST))>127).astype(np.float32)
  return torch.from_numpy(x.transpose(2,0,1).copy()),torch.from_numpy(m),int(r['class']=='tampered'),int(valid)

def infer(ds,net,dev):
 out=[];net.eval()
 with torch.no_grad():
  for x,m,y,v in DataLoader(ds,batch_size=8,shuffle=False,num_workers=2):
   z,c=net(x.to(dev));p=torch.sigmoid(z).cpu().numpy()[:,0];s=torch.sigmoid(c[:,0]).cpu().numpy()
   for i in range(len(y)):out.append({'score':float(s[i]),'label':int(y[i]),'prob':p[i],'mask':m.numpy()[i],'valid':bool(v[i])})
 return out

def loc(rows,t):
 tp=fp=fn=0;f=[];ious=[]
 for r in rows:
  if not r['valid']:continue
  a=r['prob']>=t;b=r['mask']>=.5;u=int((a&b).sum());v=int((a&~b).sum());w=int((~a&b).sum());tp+=u;fp+=v;fn+=w;f.append(2*u/max(1,2*u+v+w));ious.append(u/max(1,u+v+w))
 return {'n_images_with_masks':len(f),'pixel_f1':2*tp/max(1,2*tp+fp+fn),'pixel_iou':tp/max(1,tp+fp+fn),'image_macro_f1':float(np.mean(f)) if f else None,'image_macro_iou':float(np.mean(ious)) if ious else None}

def main():
 p=argparse.ArgumentParser();p.add_argument('--variant',choices=['rgb_only','residual_only','dual'],required=True);p.add_argument('--seed',type=int,default=None);a=p.parse_args();seed=json.loads((REPO/'configs/forenx_dualbranch_v2.json').read_text())['seed'] if a.seed is None else a.seed;folder=REPO/'results/forenx_ablation'/(a.variant if seed==20260927 else f'{a.variant}_seed{seed}');ck=torch.load(folder/'best.pt',map_location='cpu',weights_only=False);dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');net=AblationNet(a.variant).to(dev);net.load_state_dict(ck['model'])
 root=REPO/'data/external/casia_v2_raw';mf=REPO/'manifests/casia_v2_splits.csv';val=infer(EvalData(mf,root,'validation'),net,dev);test=infer(EvalData(mf,root,'test'),net,dev)
 ts=np.arange(.05,.951,.025);bestf,t=max((loc(val,float(x))['pixel_f1'],float(x)) for x in ts)
 cov=infer(EvalData(REPO/'manifests/coverage_manifest.csv',REPO/'data/external/COVERAGE'),net,dev);report={'variant':a.variant,'checkpoint_epoch':ck['epoch'],'validation_auc':score_metrics(np.array([r['label'] for r in val]),np.array([r['score'] for r in val]),.5)['roc_auc'],'casia_test_detection':score_metrics(np.array([r['label'] for r in test]),np.array([r['score'] for r in test]),.5),'casia_test_localization_threshold_from_validation':t,'casia_test_localization':loc(test,t),'coverage_zero_shot_detection':score_metrics(np.array([r['label'] for r in cov]),np.array([r['score'] for r in cov]),.5),'coverage_zero_shot_localization_threshold_from_casia_validation':t,'coverage_zero_shot_localization':loc(cov,t)}
 dest=folder/'evaluation.json';dest.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
