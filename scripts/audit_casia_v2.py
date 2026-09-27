import csv,hashlib,json
from pathlib import Path
from PIL import Image
from run_columbia_blocks_baseline import REPO
root=REPO/'data/external/casia_v2_raw'; img=root/'CASIA2'; masks=root/'CASIA2.0_Groundtruth'
def files(p): return {x.stem:x for x in p.rglob('*') if x.is_file() and x.suffix.lower() in ('.jpg','.jpeg','.tif','.tiff','.png')}
au=files(img/'Au'); tp=files(img/'Tp'); gt=files(masks)
rows=[]; missing=[]; mismatch=[]
for label,group in [('authentic',au),('tampered',tp)]:
 for stem,p in group.items():
  mask=gt.get(stem+'_gt') if label=='tampered' else None
  try:
   with Image.open(p) as im: w,h=im.size
   mh=mw=None
   if mask:
    with Image.open(mask) as m: mw,mh=m.size
    if (mw,mh)!=(w,h): mismatch.append(stem)
   elif label=='tampered': missing.append(stem)
   rows.append({'sample_id':stem,'class':label,'relative_path':p.relative_to(root).as_posix(),'mask_path':mask.relative_to(root).as_posix() if mask and (mw,mh)==(w,h) else '', 'width':w,'height':h})
  except Exception as e: missing.append(stem+':'+str(e))
out=REPO/'manifests/casia_v2_manifest.csv'
with out.open('w',newline='',encoding='utf-8') as f:
 wr=csv.DictWriter(f,fieldnames=['sample_id','class','relative_path','mask_path','width','height']);wr.writeheader();wr.writerows(rows)
summary={'authentic':len(au),'tampered':len(tp),'groundtruth_masks':len(gt),'manifest_rows':len(rows),'matched_masks':sum(bool(r['mask_path']) for r in rows),'missing_masks':len(missing),'dimension_mismatches':len(mismatch),'status':'audit_complete','missing_examples':missing[:20],'mismatch_examples':mismatch[:20]}
(root/'audit.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
