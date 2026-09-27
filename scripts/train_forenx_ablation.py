"""Train a controlled single-branch ablation on the source-connected CASIA v2 split."""
import argparse, csv, json, random
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from run_columbia_blocks_baseline import REPO
from train_forenx_dualbranch import Block

class DataV2(Dataset):
    def __init__(self, split):
        self.root=REPO/'data/external/casia_v2_raw'
        self.rows=[r for r in csv.DictReader((REPO/'manifests/casia_v2_splits.csv').open(encoding='utf-8')) if r['split']==split]
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]
        with Image.open(self.root/r['relative_path']) as im:
            x=np.asarray(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR),dtype=np.float32)/255
        x=torch.from_numpy(x.transpose(2,0,1).copy());m=torch.zeros((1,256,256));valid=bool(r['mask_path'])
        if valid:
            with Image.open(self.root/r['mask_path']) as im:
                y=np.asarray(im.convert('L').resize((256,256),Image.Resampling.NEAREST))
            m=torch.from_numpy((y>127).astype(np.float32))[None]
        return x,m,torch.tensor(float(r['class']=='tampered')),torch.tensor(float(valid))

class AblationNet(nn.Module):
    def __init__(self, variant):
        super().__init__(); self.variant=variant
        if variant in ('rgb_only','dual'):
            self.rgb=nn.Sequential(Block(3,32),Block(32,64),Block(64,128))
        if variant in ('residual_only','dual'):
            self.res=nn.Sequential(Block(3,32),Block(32,64),Block(64,128))
        channels=256 if variant=='dual' else 128
        self.fuse=nn.Sequential(nn.Conv2d(channels,128,1),nn.ReLU(True),nn.Conv2d(128,64,3,padding=1),nn.ReLU(True))
        self.maskhead=nn.Conv2d(64,1,1);self.cls=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(64,1))
    def forward(self,x):
        feat=[]
        if self.variant in ('rgb_only','dual'): feat.append(self.rgb(x))
        if self.variant in ('residual_only','dual'):
            res=x-F.avg_pool2d(F.pad(x,(1,1,1,1),mode='reflect'),3,stride=1);feat.append(self.res(res))
        z=self.fuse(torch.cat(feat,1) if len(feat)>1 else feat[0])
        return F.interpolate(self.maskhead(z),size=x.shape[-2:],mode='bilinear',align_corners=False),self.cls(z)

def auc(scores,labels):
    p=np.asarray(scores);y=np.asarray(labels,dtype=int)
    pos=p[y==1,None];neg=p[y==0]
    return float((pos>neg).mean()+.5*(pos==neg).mean())

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--variant',choices=['rgb_only','residual_only','dual'],required=True);parser.add_argument('--seed',type=int,default=None);args=parser.parse_args()
    cfg=json.loads((REPO/'configs/forenx_dualbranch_v2.json').read_text(encoding='utf-8-sig'))
    seed=cfg['seed'] if args.seed is None else args.seed;cfg['seed']=seed;torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tl=DataLoader(DataV2('train'),batch_size=cfg['batch_size'],shuffle=True,num_workers=cfg['num_workers'],pin_memory=True)
    vl=DataLoader(DataV2('validation'),batch_size=cfg['batch_size'],shuffle=False,num_workers=2,pin_memory=True)
    net=AblationNet(args.variant).to(dev);opt=torch.optim.AdamW(net.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);sc=torch.amp.GradScaler('cuda',enabled=dev.type=='cuda')
    out=REPO/'results/forenx_ablation'/f'{args.variant}_seed{seed}';out.mkdir(parents=True,exist_ok=True);best=-1;history=[]
    for ep in range(cfg['epochs']):
        net.train();losses=[]
        for x,m,y,v in tl:
            x,m,y,v=x.to(dev),m.to(dev),y.to(dev),v.to(dev);opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type,enabled=dev.type=='cuda'):
                z,c=net(x);pix=F.binary_cross_entropy_with_logits(z,m,reduction='none').mean((1,2,3));ml=(pix*v).sum()/v.sum().clamp_min(1);cl=F.binary_cross_entropy_with_logits(c[:,0],y);loss=ml+cfg['image_loss_weight']*cl
            sc.scale(loss).backward();sc.step(opt);sc.update();losses.append(float(loss.detach()))
        net.eval();scores=[];labels=[];tp=fp=fn=0
        with torch.no_grad():
            for x,m,y,v in vl:
                x,m,y,v=x.to(dev),m.to(dev),y.to(dev),v.to(dev);z,c=net(x);scores.extend(torch.sigmoid(c[:,0]).cpu().tolist());labels.extend(y.cpu().tolist());a=torch.sigmoid(z)>=.2;b=m>=.5;tp+=int((a&b).sum());fp+=int((a&~b).sum());fn+=int((~a&b).sum())
        va=auc(scores,labels);f1=2*tp/max(1,2*tp+fp+fn);row={'epoch':ep+1,'train_loss':float(np.mean(losses)),'validation_auc':va,'validation_pixel_f1_at_0.2':f1};history.append(row);print(json.dumps(row),flush=True)
        if va>best:best=va;torch.save({'model':net.state_dict(),'epoch':ep+1,'config':cfg,'variant':args.variant},out/'best.pt')
        (out/'training_log.json').write_text(json.dumps(history,indent=2)+'\n',encoding='utf-8')
    print('finished',args.variant,flush=True)
if __name__=='__main__':main()
