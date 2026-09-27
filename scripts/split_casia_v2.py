"""Create deterministic source-family-disjoint train/validation/test splits."""
import csv,hashlib,json
from collections import defaultdict
from pathlib import Path
from run_columbia_blocks_baseline import REPO

src=REPO/'manifests/casia_v2_manifest.csv'; rows=list(csv.DictReader(src.open(encoding='utf-8')))
def group(r):
 s=r['sample_id']; p=s.split('_')
 if r['class']=='authentic': return 'au:'+s.removeprefix('Au_')
 # CASIA Tp names contain manipulation family and two source identifiers;
 # grouping by the ordered source pair keeps related edits together.
 return 'tp:'+'|'.join(p[5:7]) if len(p)>7 else 'tp:'+s
groups=defaultdict(list)
for r in rows: groups[group(r)].append(r)
assign={}
for g in sorted(groups):
 b=int(hashlib.sha256(g.encode()).hexdigest()[:8],16)%100
 assign[g]='train' if b<70 else ('validation' if b<85 else 'test')
out=REPO/'manifests/casia_v2_splits.csv'
with out.open('w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0])+['source_group','split']);w.writeheader()
 for r in rows:w.writerow({**r,'source_group':group(r),'split':assign[group(r)]})
summary={'seed_rule':'sha256(source_group) mod 100; 70/15/15','groups':len(groups),'images':len(rows),'splits':{s:{'groups':sum(v==s for v in assign.values()),'images':sum(len(groups[g]) for g,v in assign.items() if v==s)} for s in ('train','validation','test')},'classes':{s:{c:sum(1 for r in rows if assign[group(r)]==s and r['class']==c) for c in ('authentic','tampered')} for s in ('train','validation','test')}}
(REPO/'data/external/casia_v2_raw/splits_audit.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
