#!/usr/bin/env python3
import json
from pathlib import Path
from PIL import Image
from run_columbia_blocks_baseline import REPO
def main():
 cfg=json.loads((REPO/'configs/trufor_stress_audit.json').read_text()); root=REPO/'data/external/columbia_uncompressed/4cam_splc'; chosen={}
 for p in sorted(root.glob('*.tif')):chosen.setdefault('_'.join(p.stem.split('_')[:2]),p)
 out=REPO/'results/trufor_stress_inputs';out.mkdir(parents=True,exist_ok=True); manifest=[]
 for pair,p in sorted(chosen.items()):
  with Image.open(p) as src: im=src.convert('RGB')
  for name,c in cfg['conditions'].items():
   q=out/f'{p.stem}__{name}.png'
   if c['type']=='jpeg':
    j=out/f'{p.stem}__{name}.jpg';im.save(j,'JPEG',quality=c['quality'],progressive=False,optimize=False); q=j
   else:
    n=(max(1,round(im.width*c['scale'])),max(1,round(im.height*c['scale'])));im.resize(n,Image.Resampling.LANCZOS).resize(im.size,Image.Resampling.BICUBIC).save(q)
   manifest.append({'sample_id':p.stem,'pair':pair,'condition':name,'path':q.relative_to(REPO).as_posix()})
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'images':len(chosen),'derivatives':len(manifest),'output':str(out)},indent=2))
if __name__=='__main__':main()
