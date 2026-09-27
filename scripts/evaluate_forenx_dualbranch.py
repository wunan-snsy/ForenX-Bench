import csv,json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from train_forenx_dualbranch import Data,Net
from run_columbia_blocks_baseline import REPO,score_metrics

def infer(split,net,device):
 ds=Data(split);loader=DataLoader(ds,batch_size=8,shuffle=False,num_workers=2);rows=[]
 net.eval()
 with torch.no_grad():
  offset=0
  for x,y,label,valid in loader:
   z=torch.sigmoid(net(x.to(device))).cpu().numpy()[:,0]; y=y.numpy()[:,0];label=label.numpy();valid=valid.numpy()
   for i in range(len(label)):
    p=z[i];ref=y[i];rows.append({'score':float(np.mean(np.sort(p.ravel())[-max(1,int(.01*p.size)):])),'label':int(label[i]),'prob':p,'mask':ref,'valid':bool(valid[i])})
 return rows
def main():
 ck=torch.load(REPO/'results/forenx_dualbranch_v0/best.pt',map_location='cpu',weights_only=False);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');net=Net().to(device);net.load_state_dict(ck['model']);val=infer('validation',net,device);test=infer('test',net,device)
 thresholds=np.arange(.05,.951,.025);best=(0,.5)
 for t in thresholds:
  tp=fp=fn=0
  for r in val:
   if not r['valid']:continue
   a=r['prob']>=t;b=r['mask']>=.5;tp+=(a&b).sum();fp+=(a&~b).sum();fn+=(~a&b).sum()
  f=2*tp/max(1,2*tp+fp+fn)
  if f>best[0]:best=(float(f),float(t))
 t=best[1];tp=fp=fn=0; per=[]
 for r in test:
  if r['valid']:
   a=r['prob']>=t;b=r['mask']>=.5;aa=int((a&b).sum());bb=int((a&~b).sum());cc=int((~a&b).sum());tp+=aa;fp+=bb;fn+=cc;per.append({'f1':2*aa/max(1,2*aa+bb+cc),'iou':aa/max(1,aa+bb+cc)})
 loc={'n_images':len(per),'pixel_f1':2*tp/max(1,2*tp+fp+fn),'pixel_iou':tp/max(1,tp+fp+fn),'image_macro_f1':float(np.mean([x['f1'] for x in per])),'image_macro_iou':float(np.mean([x['iou'] for x in per]))}
 scores=np.array([r['score'] for r in test]);labels=np.array([r['label'] for r in test]);det=score_metrics(labels,scores,.5)
 out={'best_checkpoint_epoch':ck['epoch'],'validation_threshold':t,'validation_best_pixel_f1':best[0],'test_localization':loc,'test_detection':det,'threshold_selected_on_validation_only':True}
 d=REPO/'results/forenx_dualbranch_v0';(d/'test_evaluation.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
