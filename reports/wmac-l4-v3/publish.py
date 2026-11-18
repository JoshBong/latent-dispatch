import hashlib,io,json,pathlib,shlex,subprocess,tarfile,time,traceback
ROOT=pathlib.Path('/tmp/latent-wmac-l4');DEST=ROOT/'reports/wmac-l4-v3'
BASE=['gcloud','compute','ssh','latent-wmac-l4-v3','--zone=us-central1-b','--project=deep-learning-514621','--tunnel-through-iap','--quiet']
def run(args,**kw):return subprocess.run(args,text=True,capture_output=True,timeout=300,check=True,**kw)
def ssh(cmd):return run(BASE+['--command='+cmd]).stdout
while True:
 try:
  status=ssh('cat /home/elkalt/wmac-l4.status')
  if 'exit_code=' in status:break
 except Exception as e: print('Waiting for final status:',str(e),flush=True)
 time.sleep(45)
print('Final status:',status,flush=True)
code="""import pathlib,tarfile
home=pathlib.Path('/home/elkalt');root=home/'latent-dispatch-wmac-l4'
paths=[home/'wmac-l4.status',home/'wmac-l4.log',home/'wmac-l4-requirements.txt',home/'latent-wmac-l4-run.sh',home/'wmac-l4-check.json']
archive=root/'results/wmac_gpu_v3_l4-t4.tar.gz'
paths.append(archive if archive.exists() else root/'results/wmac_gpu_v3_l4')
with tarfile.open(home/'wmac-return.tar.gz','w:gz') as tar:
 for p in paths:
  if p.exists():tar.add(p,arcname=str(p.relative_to(home)))
"""
for attempt in range(5):
 try:
  ssh('python3 -c '+shlex.quote(code))
  run(['gcloud','compute','scp','latent-wmac-l4-v3:/home/elkalt/wmac-return.tar.gz',str(DEST/'results-and-logs.tar.gz'),'--zone=us-central1-b','--project=deep-learning-514621','--tunnel-through-iap','--quiet'])
  break
 except Exception:
  if attempt==4:raise
  traceback.print_exc();time.sleep(5)
with tarfile.open(DEST/'results-and-logs.tar.gz') as tar:
 for name in ('wmac-l4.status','wmac-l4.log','wmac-l4-requirements.txt','wmac-l4-check.json'):
  try:(DEST/name).write_bytes(tar.extractfile(name).read())
  except KeyError:pass
 member='latent-dispatch-wmac-l4/results/wmac_gpu_v3_l4-t4.tar.gz'
 try:
  with tarfile.open(fileobj=io.BytesIO(tar.extractfile(member).read())) as raw:
   for m in raw.getmembers():
    if m.isfile() and (m.name.endswith('/status.json') or m.name.endswith('/summary.json')):
     data=raw.extractfile(m).read()
     safe=m.name.replace('/','__')
     (DEST/safe).write_bytes(data)
 except KeyError:pass
(DEST/'FINAL.md').write_text('# Final wrapper status\n\n```\n'+status+'```\n\nSee the archived component summaries/status for complete or partial results.\n')
(DEST/'SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name+'\n' for p in sorted(DEST.iterdir()) if p.is_file() and p.name!='SHA256SUMS'))
run(['git','add',str(DEST.relative_to(ROOT))],cwd=ROOT)
if subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT).returncode:
 run(['git','commit','-m','Save WMAC T4 final outcome, raw results, and logs'],cwd=ROOT)
for attempt in range(10):
 try:
  r=run(['git','push','origin','codex/wmac-l4-v3'],cwd=ROOT);print(r.stdout+r.stderr,flush=True);break
 except Exception:
  if attempt==9:raise
  traceback.print_exc();time.sleep(10)
run(['gcloud','compute','instances','stop','latent-wmac-l4-v3','--zone=us-central1-b','--project=deep-learning-514621','--quiet'])
print('PUBLISHED and VM stop requested.',flush=True)
