"""Dual-task CASIA v2 model: mask localization plus explicit image classification."""
import csv,json,random
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F
from run_columbia_blocks_baseline import REPO
from train_forenx_dualbranch import Block

class DataV1(Dataset):
 def __init__(self,split):
  self.root=REPO/'data/external/casia_v2_raw';self.rows=[r for r in csv.DictReader((REPO/'manifests/casia_v2_splits.csv').open(encoding='utf-8')) if r['split']==split]
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  r=self.rows[i]
  with Image.open(self.root/r['relative_path']) as im:x=np.asarray(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR),dtype=np.float32)/255
  x=torch.from_numpy(x.transpose(2,0,1).copy());mask=torch.zeros((1,256,256));valid=bool(r['mask_path'])
  if valid:
   with Image.open(self.root/r['mask_path']) as im:y=np.asarray(im.convert('L').resize((256,256),Image.Resampling.NEAREST))
   mask=torch.from_numpy((y>127).astype(np.float32))[None]
  return x,mask,torch.tensor(float(r['class']=='tampered')),torch.tensor(float(valid))
class NetV1(nn.Module):
 def __init__(self):
  super().__init__();self.rgb=nn.Sequential(Block(3,32),Block(32,64),Block(64,128));self.res=nn.Sequential(Block(3,32),Block(32,64),Block(64,128));self.fuse=nn.Sequential(nn.Conv2d(256,128,1),nn.ReLU(True),nn.Conv2d(128,64,3,padding=1),nn.ReLU(True));self.maskhead=nn.Conv2d(64,1,1);self.cls=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(64,1))
 def forward(self,x):
  res=x-F.avg_pool2d(F.pad(x,(1,1,1,1),mode='reflect'),3,stride=1);a=self.rgb(x);b=self.res(res);z=self.fuse(torch.cat([a,b],1));return F.interpolate(self.maskhead(z),size=x.shape[-2:],mode='bilinear',align_corners=False),self.cls(z)
def main():
 cfg=json.loads((REPO/'configs/forenx_dualbranch_v1.json').read_text());torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);random.seed(cfg['seed']);dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');tl=DataLoader(DataV1('train'),batch_size=8,shuffle=True,num_workers=4,pin_memory=True);vl=DataLoader(DataV1('validation'),batch_size=8,shuffle=False,num_workers=2,pin_memory=True);net=NetV1().to(dev);opt=torch.optim.AdamW(net.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);sc=torch.amp.GradScaler('cuda',enabled=dev.type=='cuda');out=REPO/'results/forenx_dualbranch_v1';out.mkdir(parents=True,exist_ok=True);best=-1;history=[]
 for ep in range(20):
  net.train();ls=[]
  for x,m,y,v in tl:
   x,m,y,v=x.to(dev),m.to(dev),y.to(dev),v.to(dev);opt.zero_grad(set_to_none=True)
   with torch.autocast(device_type=dev.type,enabled=dev.type=='cuda'):
    z,c=net(x);pl=F.binary_cross_entropy_with_logits(z,m,reduction='none').mean((1,2,3));ml=(pl*v).sum()/v.sum().clamp_min(1);cl=F.binary_cross_entropy_with_logits(c[:,0],y);loss=ml+.5*cl
   sc.scale(loss).backward();sc.step(opt);sc.update();ls.append(float(loss.detach()))
  net.eval();pred=[];truth=[];tp=fp=fn=0
  with torch.no_grad():
   for x,m,y,v in vl:
    x,m,y,v=x.to(dev),m.to(dev),y.to(dev),v.to(dev);z,c=net(x);pred.extend(torch.sigmoid(c[:,0]).cpu().tolist());truth.extend(y.cpu().tolist());a=torch.sigmoid(z)>=.2;b=m>=.5;tp+=int((a&b).sum());fp+=int((a&~b).sum());fn+=int((~a&b).sum())
  # Group-independent fixed split; report AUC via rank statistic.
  p=np.array(pred);q=np.array(truth);auc=float(((p[q==1,None]>p[q==0]).mean()+.5*(p[q==1,None]==p[q==0]).mean()));f1=2*tp/max(1,2*tp+fp+fn);rec={'epoch':ep+1,'train_loss':float(np.mean(ls)),'validation_auc':auc,'validation_mask_f1_at_0.2':f1};history.append(rec);print(json.dumps(rec),flush=True)
  if auc>best:best=auc;torch.save({'model':net.state_dict(),'epoch':ep+1,'config':cfg},out/'best.pt')
  (out/'training_log.json').write_text(json.dumps(history,indent=2)+'\n')
 print('finished',flush=True)
if __name__=='__main__':main()
