"""External zero-shot evaluation on COVERAGE using the CASIA-trained checkpoint."""
import csv, json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from run_columbia_blocks_baseline import REPO, score_metrics
from train_forenx_dualbranch_v1 import NetV1

class Coverage(Dataset):
    def __init__(self):
        self.root=REPO/'data/external/COVERAGE'
        self.rows=list(csv.DictReader((REPO/'manifests/coverage_manifest.csv').open(encoding='utf-8')))
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]
        with Image.open(self.root/r['relative_path']) as im:
            x=np.asarray(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR),dtype=np.float32)/255
        x=torch.from_numpy(x.transpose(2,0,1).copy())
        valid=bool(r['mask_path']); mask=np.zeros((256,256),dtype=np.float32)
        if valid:
            with Image.open(self.root/r['mask_path']) as im:
                mask=(np.asarray(im.convert('L').resize((256,256),Image.Resampling.NEAREST))>127).astype(np.float32)
        return x,torch.from_numpy(mask),int(r['class']=='tampered'),int(valid)

def main():
    checkpoint=REPO/'results/forenx_dualbranch_v2_source_group/best.pt'
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');net=NetV1().to(dev);net.load_state_dict(ck['model']);net.eval()
    loader=DataLoader(Coverage(),batch_size=8,shuffle=False,num_workers=2)
    rows=[]
    with torch.no_grad():
        for x,m,y,v in loader:
            z,c=net(x.to(dev));p=torch.sigmoid(z).cpu().numpy()[:,0];s=torch.sigmoid(c[:,0]).cpu().numpy()
            for i in range(len(y)): rows.append((float(s[i]),int(y[i]),p[i],m.numpy()[i],bool(v[i])))
    threshold=json.loads((REPO/'results/forenx_dualbranch_v2_source_group/test_evaluation.json').read_text())['validation_localization_threshold']
    labels=np.array([r[1] for r in rows]);scores=np.array([r[0] for r in rows]);det=score_metrics(labels,scores,.5)
    tp=fp=fn=0;per=[];nmask=0
    for score,label,p,m,valid in rows:
        if not valid: continue
        nmask+=1; a=p>=threshold;b=m>=.5;u=int((a&b).sum());v=int((a&~b).sum());w=int((~a&b).sum());tp+=u;fp+=v;fn+=w;per.append((2*u/max(1,2*u+v+w),u/max(1,u+v+w)))
    out={'evaluation':'zero-shot external evaluation; no COVERAGE training or threshold tuning','checkpoint_epoch':ck['epoch'],'n_images':len(rows),'paired_samples':len(rows)//2,'detection':det,'localization_threshold_from_CASIA_validation':threshold,'localization':{'n_images_with_masks':nmask,'pixel_f1':2*tp/max(1,2*tp+fp+fn),'pixel_iou':tp/max(1,tp+fp+fn),'image_macro_f1':float(np.mean([x[0] for x in per])),'image_macro_iou':float(np.mean([x[1] for x in per]))}}
    dest=REPO/'results/forenx_dualbranch_v2_source_group/coverage_zero_shot_evaluation.json';dest.write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
