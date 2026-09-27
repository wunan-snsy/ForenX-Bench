#!/usr/bin/env python3
"""Stratified independent-process replay with distinct RSR and EGA metrics."""
import argparse,csv,json,random,subprocess,sys,hashlib
from pathlib import Path
import numpy as np
from run_columbia_blocks_baseline import REPO
from evaluate_casia_v1_models import prediction_path


def sha_file(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def wilson(k,n,z=1.959963984540054):
 if not n:return [None,None]
 p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;r=z*((p*(1-p)/n+z*z/(4*n*n))**.5)/d
 return [max(0.,c-r),min(1.,c+r)]

def summarize(records):
 attempts=len(records);successful=sum(r['replay_succeeded'] for r in records);agreed=sum(r.get('evidence_agrees',False) for r in records if r['replay_succeeded'])
 return {'attempted':attempts,'successful_outputs':successful,'RSR':successful/attempts if attempts else None,'RSR_wilson95':wilson(successful,attempts),'agreement_within_tolerance':agreed,'EGA':agreed/successful if successful else None,'EGA_wilson95':wilson(agreed,successful)}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--per-stratum',type=int,default=30);ap.add_argument('--seed',type=int,default=20260929);ap.add_argument('--limit-models',nargs='*',choices=['trufor','mvss'],default=['trufor','mvss']);ap.add_argument('--resume',action='store_true');args=ap.parse_args()
 with (REPO/'manifests/casia_v1_manifest.csv').open(encoding='utf-8') as f:rows=list(csv.DictReader(f))
 strata={'authentic':[],'CM':[],'Sp':[]}
 for r in rows:
  group='authentic' if r['class']=='authentic' else r['manipulation']
  if group in strata:strata[group].append(r)
 rng=random.Random(args.seed);chosen=[]
 for name,pool in strata.items():
  if len(pool)<args.per_stratum:raise ValueError(f'{name}: requested {args.per_stratum}, only {len(pool)} available')
  chosen.extend((name,r) for r in rng.sample(sorted(pool,key=lambda x:x['sample_id']),args.per_stratum))
 root=REPO/'data/external/casia_v1_corrected';out=REPO/'results/casia_v1_replay';out.mkdir(parents=True,exist_ok=True);records=[];checkpoint=out/'summary_v1.1.progress.jsonl'
 if args.resume and checkpoint.exists():
  for line in checkpoint.read_text(encoding='utf-8').splitlines():
   try: records.append(json.loads(line))
   except json.JSONDecodeError: pass
 done={(r['model'],r['sample_id']) for r in records}
 if args.resume and not checkpoint.exists():
  for model in args.limit_models:
   for stratum,r in chosen:
    image=root/r['relative_path'];path=out/model/'stratified_v1_1'/r['sample_id']/(image.name+'.npz')
    if not path.is_file() or sha_file(image)!=r['sha256']: continue
    try:
     original=np.load(prediction_path(model,r));z=np.load(path);same=np.asarray(z['map']).shape==np.asarray(original['map']).shape
     se=float(abs(float(z['score'])-float(original['score'])));me=float(np.abs(np.asarray(z['map'])-np.asarray(original['map'])).max()) if same else None
     item={'model':model,'sample_id':r['sample_id'],'stratum':stratum,'input_sha256_matches_manifest':True,'recovered_existing_output':True,'replay_succeeded':True,'returncode':0,'score_abs_error':se,'map_max_abs_error':me,'map_shape_matches':same,'evidence_agrees':same and se<=1e-5 and me<=1e-5}
     records.append(item);done.add((model,r['sample_id']))
    except Exception: pass
  with checkpoint.open('w',encoding='utf-8') as f:
   for item in records:f.write(json.dumps(item)+'\n')
 pytrufor=REPO/'external/TruFor/.venv/Scripts/python.exe';cwdtrufor=REPO/'external/TruFor/test_docker/src'
 for model in args.limit_models:
  for stratum,r in chosen:
   if (model,r['sample_id']) in done: continue
   image=root/r['relative_path'];original=np.load(prediction_path(model,r));temp=out/model/'stratified_v1_1'/r['sample_id'];temp.mkdir(parents=True,exist_ok=True)
   item={'model':model,'sample_id':r['sample_id'],'stratum':stratum,'input_sha256_matches_manifest':sha_file(image)==r['sha256']}
   if not item['input_sha256_matches_manifest']:
    item.update({'replay_succeeded':False,'evidence_agrees':False,'error':'input hash differs from manifest'});records.append(item);done.add((model,r['sample_id']))
    with checkpoint.open('a',encoding='utf-8') as f:f.write(json.dumps(item)+'\n')
    continue
   if model=='trufor':
    command=[str(pytrufor),'trufor_test.py','-gpu','0','-in',str(image),'-out',str(temp)];cwd=cwdtrufor
   else:
    command=[sys.executable,str(REPO/'scripts/run_mvss_inference.py'),'--input',str(image),'--output',str(temp),'--gpu','0'];cwd=REPO
   try:
    run=subprocess.run(command,cwd=cwd,capture_output=True,text=True)
    replay_path=temp/(image.name+'.npz');ok=run.returncode==0 and replay_path.is_file();item['returncode']=run.returncode;item['replay_succeeded']=ok
    if ok:
     z=np.load(replay_path);same_shape=np.asarray(z['map']).shape==np.asarray(original['map']).shape
     item['score_abs_error']=float(abs(float(z['score'])-float(original['score'])))
     item['map_max_abs_error']=float(np.abs(np.asarray(z['map'])-np.asarray(original['map'])).max()) if same_shape else None
     item['map_shape_matches']=same_shape
     item['evidence_agrees']=same_shape and item['score_abs_error']<=1e-5 and item['map_max_abs_error']<=1e-5
    else:
     item.update({'evidence_agrees':False,'stderr':run.stderr[-500:]})
   except Exception as e:item.update({'replay_succeeded':False,'evidence_agrees':False,'error':str(e)})
   records.append(item);done.add((model,r['sample_id']))
   with checkpoint.open('a',encoding='utf-8') as f:f.write(json.dumps(item)+'\n')
   print(json.dumps(item),flush=True)
 by_model={m:summarize([r for r in records if r['model']==m]) for m in args.limit_models}
 by_stratum={m:{g:summarize([r for r in records if r['model']==m and r['stratum']==g]) for g in strata} for m in args.limit_models}
 result={'status':'complete' if len(records)==len(chosen)*len(args.limit_models) else 'incomplete','schema_version':'forenx-replay-1.1','selection':{'method':'seeded random sample within authentic, copy-move, and splicing strata','per_stratum_per_model':args.per_stratum,'seed':args.seed},'tolerance':{'score_abs_error':1e-5,'map_max_abs_error':1e-5},'metric_definitions':{'RSR':'successful subprocess output artifacts divided by attempted replays','EGA':'successful replay outputs agreeing with original score and map within tolerance divided by successful replays'},'by_model':by_model,'by_stratum':by_stratum,'records':records}
 (out/'summary_v1.1.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps({'status':result['status'],'by_model':by_model,'by_stratum':by_stratum},indent=2))
if __name__=='__main__':main()
