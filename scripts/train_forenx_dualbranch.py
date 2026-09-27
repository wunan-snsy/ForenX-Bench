"""Initial ForenX dual-branch RGB/residual localization model on CASIA v2."""
import csv,json,random,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F
from run_columbia_blocks_baseline import REPO

class Data(Dataset):
 def __init__(self,split):
  self.root=REPO/'data/external/casia_v2_raw';self.rows=[r for r in csv.DictReader((REPO/'manifests/casia_v2_splits.csv').open(encoding='utf-8')) if r['split']==split]
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  r=self.rows[i];p=self.root/r['relative_path']
  with Image.open(p) as im:x=np.asarray(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR),dtype=np.float32)/255
  x=torch.from_numpy(x.transpose(2,0,1).copy()); mask=torch.zeros((1,256,256),dtype=torch.float32); valid=bool(r['mask_path'])
  if valid:
   with Image.open(self.root/r['mask_path']) as im:y=np.asarray(im.convert('L').resize((256,256),Image.Resampling.NEAREST))
   mask=torch.from_numpy((y>127).astype(np.float32))[None]
  label=torch.tensor(float(r['class']=='tampered'));return x,mask,label,torch.tensor(valid,dtype=torch.float32)

class Block(nn.Module):
 def __init__(self,a,b):super().__init__();self.net=nn.Sequential(nn.Conv2d(a,b,3,padding=1,stride=2,bias=False),nn.BatchNorm2d(b),nn.ReLU(inplace=True),nn.Conv2d(b,b,3,padding=1,bias=False),nn.BatchNorm2d(b),nn.ReLU(inplace=True))
 def forward(self,x):return self.net(x)
class Net(nn.Module):
 def __init__(self):
  super().__init__();self.rgb=nn.Sequential(Block(3,32),Block(32,64),Block(64,128));self.res=nn.Sequential(Block(3,32),Block(32,64),Block(64,128));self.fuse=nn.Sequential(nn.Conv2d(256,128,1),nn.ReLU(inplace=True),nn.Conv2d(128,64,3,padding=1),nn.ReLU(inplace=True));self.head=nn.Conv2d(64,1,1)
 def forward(self,x):
  # Fixed high-pass residual branch; reflect padding avoids border-size changes.
  low=F.avg_pool2d(F.pad(x,(1,1,1,1),mode='reflect'),3,stride=1);res=x-low
  a=self.rgb(x);b=self.res(res);z=self.fuse(torch.cat([a,b],1));return F.interpolate(self.head(z),size=x.shape[-2:],mode='bilinear',align_corners=False)
def main():
 cfg=json.loads((REPO/'configs/forenx_dualbranch_v0.json').read_text());torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);random.seed(cfg['seed']);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 train=Data('train');val=Data('validation');tl=DataLoader(train,batch_size=cfg['batch_size'],shuffle=True,num_workers=cfg['num_workers'],pin_memory=True);vl=DataLoader(val,batch_size=cfg['batch_size'],shuffle=False,num_workers=cfg['num_workers'],pin_memory=True)
 net=Net().to(device);opt=torch.optim.AdamW(net.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda' and cfg['mixed_precision']);best=-1;out=REPO/'results/forenx_dualbranch_v0';out.mkdir(parents=True,exist_ok=True);log=[]
 for epoch in range(cfg['epochs']):
  net.train();losses=[]
  for x,y,label,valid in tl:
   x=x.to(device);y=y.to(device);valid=valid.to(device);opt.zero_grad(set_to_none=True)
   with torch.autocast(device_type=device.type,enabled=device.type=='cuda' and cfg['mixed_precision']):
    z=net(x);pix=F.binary_cross_entropy_with_logits(z,y,reduction='none').mean((1,2,3));loss=(pix*valid).sum()/valid.sum().clamp_min(1)
   scaler.scale(loss).backward();scaler.step(opt);scaler.update();losses.append(float(loss.detach()))
  net.eval();tp=fp=fn=0;vls=[]
  with torch.no_grad():
   for x,y,label,valid in vl:
    x=x.to(device);y=y.to(device);valid=valid.to(device);z=net(x);pred=torch.sigmoid(z)>=.5;ref=y>=.5;tp+=int((pred&ref).sum());fp+=int((pred&~ref).sum());fn+=int((~pred&ref).sum());vls.append(float((F.binary_cross_entropy_with_logits(z,y,reduction='none').mean((1,2,3))*valid).sum()/valid.sum().clamp_min(1)))
  f1=2*tp/max(1,2*tp+fp+fn);entry={'epoch':epoch+1,'train_mask_loss':float(np.mean(losses)),'validation_mask_loss':float(np.mean(vls)),'validation_pixel_f1':f1};log.append(entry);print(json.dumps(entry),flush=True)
  if f1>best:best=f1;torch.save({'model':net.state_dict(),'epoch':epoch+1,'config':cfg},out/'best.pt')
  (out/'training_log.json').write_text(json.dumps(log,indent=2)+'\n')
 print('finished',out,flush=True)
if __name__=='__main__':main()
