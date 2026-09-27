#!/usr/bin/env python3
"""Create resumable COVERAGE derivatives under the frozen CASIA stress protocol."""
import csv, json
from pathlib import Path
from PIL import Image
from run_columbia_blocks_baseline import REPO

def main():
    cfg=json.loads((REPO/'configs/casia_v1_full_stress.json').read_text(encoding='utf-8'))
    root=REPO/'data/external/COVERAGE'; out=REPO/'results/coverage_full_stress_inputs'
    with (REPO/'manifests/coverage_manifest.csv').open(encoding='utf-8') as h: rows=list(csv.DictReader(h))
    manifest=[]; made=reused=0
    for row in rows:
        for condition,spec in cfg['conditions'].items():
            target=out/condition/row['class']/(row['sample_id']+spec['extension']); target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists() and target.stat().st_size: reused+=1
            else:
                with Image.open(root/row['relative_path']) as source:
                    image=source.convert('RGB')
                    if spec['type']=='jpeg': image.save(target,'JPEG',quality=spec['quality'],progressive=False,optimize=False)
                    else:
                        size=(max(1,round(image.width*spec['scale'])),max(1,round(image.height*spec['scale'])))
                        image.resize(size,Image.Resampling.LANCZOS).resize(image.size,Image.Resampling.BICUBIC).save(target,'JPEG',quality=spec['quality'],progressive=False,optimize=False)
                made+=1
            manifest.append({**{k:row[k] for k in ('sample_id','class','pair_id','relative_path','mask_path')},'condition':condition,'path':target.relative_to(REPO).as_posix()})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'expected':800,'created':made,'reused':reused},indent=2))
if __name__=='__main__':main()
