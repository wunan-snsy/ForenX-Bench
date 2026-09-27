"""Zero-shot image-level evaluation on Columbia 4-camera set.

Only image-level detection is evaluated: available Columbia masks annotate
boundary/source regions and are not complete generic tampering masks.
"""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset,DataLoader
from run_columbia_blocks_baseline import REPO,score_metrics
from train_forenx_ablation import AblationNet

class Columbia(Dataset):
 def __init__(self):
  root=REPO/'data/external/columbia_uncompressed';self.rows=[]
  for cls,folder in [('authentic','4cam_auth'),('tampered','4cam_splc')]:
   for p in sorted((root/folder).glob('*.tif')):self.rows.append((p,cls))
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  p,cls=self.rows[i]
  with Image.open(p) as im:x=np.asarray(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR),dtype=np.float32)/255
  return torch.from_numpy(x.transpose(2,0,1).copy()),int(cls=='tampered'),str(p.relative_to(REPO/'data/external/columbia_uncompressed'))

def get_checkpoint(variant,seed):
 if seed==20260927 and variant=='dual':return REPO/'results/forenx_dualbranch_v2_source_group/best.pt'
 suffix='' if seed==20260927 else f'_seed{seed}'
 return REPO/'results/forenx_ablation'/f'{variant}{suffix}'/'best.pt'

def main():
 dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');ds=Columbia();loader=DataLoader(ds,batch_size=8,shuffle=False,num_workers=2);results=[]
 for variant in ('residual_only','dual'):
  for seed in (20260927,20260928):
   path=get_checkpoint(variant,seed);ck=torch.load(path,map_location='cpu',weights_only=False);net=AblationNet(variant).to(dev);net.load_state_dict(ck['model']);net.eval();ys=[];ss=[];names=[]
   with torch.no_grad():
    for x,y,n in loader:
     _,c=net(x.to(dev));ss.extend(torch.sigmoid(c[:,0]).cpu().tolist());ys.extend(y.tolist());names.extend(n)
   metrics=score_metrics(np.array(ys),np.array(ss),.5)
   results.append({'variant':variant,'seed':seed,'checkpoint_epoch':int(ck['epoch']),'n_images':len(ys),'authentic':int((np.array(ys)==0).sum()),'tampered':int((np.array(ys)==1).sum()),'zero_shot_detection':metrics,'threshold_tuning':'none; fixed score threshold 0.5 reported for accuracy'})
   pred_path=REPO/'results/forenx_ablation'/f'columbia_{variant}_seed{seed}_predictions.csv'
   import csv
   with pred_path.open('w',newline='',encoding='utf-8') as f:
    w=csv.writer(f);w.writerow(['relative_path','label','tamper_probability']);w.writerows(zip(names,ys,ss))
 out={'dataset':'Columbia 4-camera authentic/spliced image-level evaluation','input_resize':'RGB BILINEAR to 256 x 256; same as training','mask_use':'none; mask is not a full-image tampering reference','models':results}
 dest=REPO/'results/forenx_ablation/columbia_zero_shot_summary.json';dest.write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
