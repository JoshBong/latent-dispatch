"""Retrieve this run by SSH or checksum-framed Cloud Logging backup, analyze, push, stop."""
import base64,hashlib,importlib.util,json,pathlib,re,subprocess,sys,time,traceback
ROOT=pathlib.Path('/tmp/latent-wmac-4b-together');HERE=ROOT/'reports/wmac-4b-together'
VM='latent-wmac-l4-v3';ZONE='us-central1-b';PROJECT='deep-learning-514621'
CONFIG=json.loads((HERE/'instance-run.json').read_text());START=CONFIG['lastStartTimestamp'];ID=CONFIG['id']
s=importlib.util.spec_from_file_location('artifact',HERE/'artifact.py');artifact=importlib.util.module_from_spec(s);s.loader.exec_module(artifact)
def run(args,timeout=90):return subprocess.run(args,text=True,capture_output=True,timeout=timeout,check=True)
def ssh(cmd):return run(['gcloud','compute','ssh',VM,'--zone='+ZONE,'--project='+PROJECT,'--tunnel-through-iap','--quiet','--command='+cmd]).stdout
out=HERE/'results-and-logs.tar.gz';transport=None
while transport is None:
    try:
        check=ssh("state=$(systemctl show latent-wmac-4b-together --property=ActiveState --value); case $state in inactive|failed) ;; *) exit 3;; esac; grep -q exit_code= /home/elkalt/wmac-4b-together.status && gzip -t /home/elkalt/wmac-4b-together-results.tar.gz && sha256sum /home/elkalt/wmac-4b-together-results.tar.gz")
        digest=check.split()[0]
        run(['gcloud','compute','scp',VM+':/home/elkalt/wmac-4b-together-results.tar.gz',str(out),'--zone='+ZONE,'--project='+PROJECT,'--tunnel-through-iap','--quiet'])
        assert hashlib.sha256(out.read_bytes()).hexdigest()==digest
        transport='ssh'
    except Exception:
        try:
            query=f'resource.type="gce_instance" AND resource.labels.instance_id="{ID}" AND timestamp>="{START}" AND log_id("serialconsole.googleapis.com/serial_port_1_output")'
            r=run(['gcloud','logging','read',query,'--project='+PROJECT,'--order=asc','--limit=5000','--format=json'])
            text=''.join(x.get('textPayload','') for x in json.loads(r.stdout))
            if 'experiment=wmac-4b-together' not in text:raise ValueError('No outcome marker yet')
            payload=artifact.decode(text);out.write_bytes(payload);digest=hashlib.sha256(payload).hexdigest();transport='cloud_serial_log'
        except Exception:
            time.sleep(45)
print('Retrieved checksum-valid control artifact via',transport,flush=True)
(HERE/'retrieval.json').write_text(json.dumps(dict(transport=transport,sha256=digest,bytes=out.stat().st_size,instance_id=ID,since=START),indent=2)+'\n')
result=subprocess.run(['/tmp/latent-wmac-venv/bin/python',str(HERE/'analyze.py')],cwd=ROOT,text=True,capture_output=True)
if result.returncode:
    (HERE/'analysis-error.txt').write_text(result.stdout+result.stderr)
    print('Partial/final archive retrieved; analysis needs inspection.',flush=True)
else:
    (HERE/'analysis-error.txt').unlink(missing_ok=True)
(HERE/'SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(HERE))+'\n' for p in sorted(HERE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name!='SHA256SUMS'))
run(['git','-C',str(ROOT),'add',str(HERE.relative_to(ROOT))])
if subprocess.run(['git','-C',str(ROOT),'diff','--cached','--quiet']).returncode:
    run(['git','-C',str(ROOT),'commit','-m','Save 4B FP16 same-worker control outcome and paired analysis'])
for attempt in range(10):
    try:run(['git','-C',str(ROOT),'push','origin','codex/wmac-4b-together']);break
    except Exception:
        if attempt==9:raise
        time.sleep(15)
print('PUBLISHED',flush=True)
run(['gcloud','compute','instances','stop',VM,'--zone='+ZONE,'--project='+PROJECT,'--quiet'],timeout=180)
print('VM stop requested.',flush=True)
