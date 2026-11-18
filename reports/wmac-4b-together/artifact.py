"""Small, checksummed control-result backup through the already enabled serial log."""
import argparse,base64,hashlib,io,json,pathlib,re,subprocess,tarfile
PREFIX='WMAC_CTRL_ARTIFACT'
RX=re.compile(PREFIX+r' sha=([0-9a-f]{64}) i=(\d+) n=(\d+) bytes=(\d+) data=([A-Za-z0-9+/=]+)')
def frames(payload):
    sha=hashlib.sha256(payload).hexdigest();width=2304;n=(len(payload)+width-1)//width
    for i in range(n):
        data=base64.b64encode(payload[i*width:(i+1)*width]).decode()
        yield f'{PREFIX} sha={sha} i={i} n={n} bytes={len(payload)} data={data}\n'
def decode(text):
    groups={}
    for sha,i,n,size,data in RX.findall(text):
        g=groups.setdefault(sha,dict(n=int(n),size=int(size),parts={}))
        assert g['n']==int(n) and g['size']==int(size)
        chunk=base64.b64decode(data,validate=True);i=int(i)
        if i in g['parts']:assert g['parts'][i]==chunk
        g['parts'][i]=chunk
    for sha,g in reversed(list(groups.items())):
        if set(g['parts'])!=set(range(g['n'])):continue
        payload=b''.join(g['parts'][i] for i in range(g['n']))
        if len(payload)==g['size'] and hashlib.sha256(payload).hexdigest()==sha:return payload
    raise ValueError('No complete checksum-valid control artifact in serial logs yet')
def pack(home,checkout):
    paths=[home/'wmac-4b-together',home/'wmac-4b-together-smoke',home/'wmac-4b-together.log',
           home/'wmac-4b-together.status',home/'wmac-4b-together-requirements.txt',home/'run-wmac-4b-together.sh']
    out=home/'wmac-4b-together-results.tar.gz'
    with tarfile.open(out,'w:gz') as tar:
        for p in paths:
            if p.exists():tar.add(p,arcname=str(p.relative_to(home)))
        tar.add(checkout/'reports/wmac-4b-together/wmac_together.py',arcname='runtime/wmac_together.py')
        tar.add(checkout/'latent_dispatch',arcname='runtime/latent_dispatch',filter=lambda m:None if '__pycache__' in m.name else m)
    payload=out.read_bytes()
    print(json.dumps(dict(archive=str(out),bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest())),flush=True)
    return payload
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['pack','selftest']);p.add_argument('--home',type=pathlib.Path);p.add_argument('--checkout',type=pathlib.Path);p.add_argument('--serial',action='store_true');a=p.parse_args()
    if a.command=='selftest':
        payload=bytes(range(256))*400;parts=list(frames(payload))
        assert decode(''.join(reversed(parts))+parts[0])==payload
        try:decode(''.join(parts[:-1]));raise AssertionError('accepted missing chunk')
        except ValueError:pass
        print('artifact roundtrip, duplicates, and missing-chunk checks passed')
    else:
        payload=pack(a.home,a.checkout)
        if a.serial:
            proc=subprocess.Popen(['sudo','tee','/dev/ttyS0'],stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,text=True)
            for line in frames(payload):proc.stdin.write(line)
            proc.stdin.close();assert proc.wait()==0
            print('Checksum-framed control backup written to serial port',flush=True)
