"""Download official Visual Coding NWB sessions (no LFP/raw AP), resumably.

Metadata and data live on the external drive; acquisition never inspects outcomes.
"""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import os
import sys
import time
import xml.etree.ElementTree as ET

import requests

ROOT = Path('/Volumes/MimiHDD/energy_nonnormal_data/allen_visual_coding')
BASE = 'https://allen-brain-observatory.s3.us-west-2.amazonaws.com/'
PREFIX = 'visual-coding-neuropixels/ecephys-cache/'

def manifest(root):
    root.mkdir(parents=True, exist_ok=True)
    records = []
    token = None
    while True:
        params = {'list-type': '2', 'prefix': PREFIX, 'max-keys': 1000}
        if token:
            params['continuation-token'] = token
        r = requests.get(BASE, params=params, timeout=90)
        r.raise_for_status()
        doc = ET.fromstring(r.content)
        ns = {'s': 'http://s3.amazonaws.com/doc/2006-03-01/'}
        for obj in doc.findall('s:Contents', ns):
            key = obj.findtext('s:Key', namespaces=ns)
            records.append({'key': key, 'bytes': int(obj.findtext('s:Size', namespaces=ns)),
                            'etag': obj.findtext('s:ETag', namespaces=ns),
                            'last_modified': obj.findtext('s:LastModified', namespaces=ns)})
        token = doc.findtext('s:NextContinuationToken', namespaces=ns)
        if not token:
            break
    (root / 's3_inventory.json').write_text(json.dumps(records, indent=2))
    sessions = [x for x in records if '/session_' in x['key'] and x['key'].endswith('.nwb')
                and '/probe_' not in x['key'] and '_probe_' not in x['key']]
    (root / 'download_manifest.json').write_text(json.dumps(sorted(sessions, key=lambda x:x['key']), indent=2))
    for name in ('sessions.csv', 'units.csv', 'channels.csv', 'probes.csv', 'manifest.json'):
        if (root / name).exists():
            continue
        r = requests.get(BASE + PREFIX + name, timeout=120)
        r.raise_for_status()
        (root / name).write_bytes(r.content)
    sessions = [x for x in records if '/session_' in x['key'] and x['key'].endswith('.nwb')
                and '/probe_' not in x['key'] and '_probe_' not in x['key']]
    (root / 'download_manifest.json').write_text(json.dumps(sorted(sessions, key=lambda x:x['key']), indent=2))
    print('sessions', len(sessions), 'GiB', sum(x['bytes'] for x in sessions)/2**30, flush=True)
    print(json.dumps(sessions[:2], indent=2), flush=True)

def download(entry, root, connections=4):
    dest = root / 'nwb' / Path(entry['key']).name
    dest.parent.mkdir(exist_ok=True)
    partial = dest.with_suffix('.nwb.partial')
    report = dest.with_suffix('.nwb.provenance.json')
    if dest.exists() and dest.stat().st_size == entry['bytes'] and report.exists():
        return {'file': str(dest), 'status': 'already_verified'}
    start = time.time()
    url = BASE + entry['key']
    # Each range has its own URL cache key. Never trust a short 206 as a full NWB.
    chunks = root / 'chunks_512k' / dest.stem
    chunks.mkdir(parents=True, exist_ok=True)
    block = 512*1024
    def fetch(i):
        if not Path('/Volumes/MimiHDD').is_mount():
            raise RuntimeError('MimiHDD disconnected')
        lo, hi = i*block, min((i+1)*block, entry['bytes'])-1
        p = chunks / f'{i:06d}.part'
        if p.exists() and p.stat().st_size == hi-lo+1:
            return p
        for attempt in range(6):
            try:
                # curl max-time enforces wall time even for trickling connections.
                tmp, hdr = p.with_suffix('.tmp'), p.with_suffix('.headers')
                subprocess.run(['curl','--fail','--location','--silent','--show-error',
                    '--connect-timeout','25','--max-time','150','--range',f'{lo}-{hi}',
                    '--dump-header',str(hdr),'--output',str(tmp),
                    url+f'?download_range={lo}-{hi}'],check=True)
                headers=hdr.read_text().lower()
                if f'content-range: bytes {lo}-{hi}/{entry["bytes"]}' not in headers:
                    raise RuntimeError('Invalid Content-Range')
                if entry['etag'].lower() not in headers:
                    raise RuntimeError('Object ETag differs from frozen inventory')
                if tmp.stat().st_size != hi-lo+1:
                    raise RuntimeError('Truncated range')
                tmp.replace(p)
                return p
            except Exception:
                if attempt==5: raise
                time.sleep(min(5*(attempt+1),30))
    count=(entry['bytes']+block-1)//block
    with concurrent.futures.ThreadPoolExecutor(max_workers=connections) as pool:
        futures=[pool.submit(fetch,i) for i in range(count)]
        for n, future in enumerate(concurrent.futures.as_completed(futures),1):
            future.result()
            if n%32==0 or n==count:
                print(json.dumps({'session':dest.stem,'verified_chunks':n,'total_chunks':count}),flush=True)
    with partial.open('wb') as out:
        for i in range(count):
            with (chunks/f'{i:06d}.part').open('rb') as src:
                shutil.copyfileobj(src,out,8*1024**2)
        out.flush()
        os.fsync(out.fileno())
    if partial.stat().st_size != entry['bytes']:
        raise RuntimeError(f'Size mismatch {partial}')
    digest = hashlib.sha256()
    with partial.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2), b''):
            digest.update(block)
    import h5py
    with h5py.File(partial, 'r') as f:
        if 'units' not in f:
            raise RuntimeError(f'No units table in {partial}')
    partial.rename(dest)
    result = {**entry, 'url': url, 'sha256': digest.hexdigest(), 'elapsed_s': time.time()-start,
              'status': 'downloaded_size_and_hdf5_verified', 'file': str(dest)}
    report.write_text(json.dumps(result, indent=2))
    # Only generated chunk files are removed after full-file verification.
    for p in chunks.iterdir():
        # exFAT/macOS AppleDouble companions can disappear with the data fork.
        if not p.name.startswith('.') and p.suffix in ('.part','.headers','.tmp'):
            try:
                p.unlink()
            except FileNotFoundError:
                pass
    return result

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--manifest-only', action='store_true')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--connections', type=int, default=4,
                   help='Concurrent validated range requests per session')
    p.add_argument('--analyze', action='store_true')
    a = p.parse_args()
    if not a.root.parent.exists():
        raise RuntimeError('External drive must be mounted; refusing internal fallback')
    if not (a.root/'download_manifest.json').exists():
        manifest(a.root)
    if a.manifest_only:
        return
    entries = json.loads((a.root/'download_manifest.json').read_text())
    if a.limit:
        entries = entries[:a.limit]
    needed=0
    for x in entries:
        name=Path(x['key']).name
        done=a.root/'nwb'/name
        if done.exists() and done.with_suffix('.nwb.provenance.json').exists():
            continue
        stored=sum(p.stat().st_size for p in (a.root/'chunks_512k'/Path(name).stem).glob('*.part') if not p.name.startswith('.'))
        needed+=max(0,x['bytes']-stored)
    reserve=30*2**30 + a.workers*max(x['bytes'] for x in entries)
    if needed + reserve > shutil.disk_usage(a.root).free:
        raise RuntimeError('Insufficient free space preserving 30 GiB reserve')
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures = {pool.submit(download, e, a.root, a.connections): e for e in entries}
        failures = []
        for future in concurrent.futures.as_completed(futures):
            try:
                r = future.result()
                print(json.dumps(r), flush=True)
                if a.analyze:
                    logdir=a.root/'analysis_logs';logdir.mkdir(exist_ok=True)
                    with (logdir/(Path(r['file']).stem+'.log')).open('w') as log:
                        child=subprocess.run([sys.executable,str(Path(__file__).with_name('analyze_session.py')),
                            r['file'],'--out',str(a.root/'analysis')],stdout=log,stderr=subprocess.STDOUT)
                    print(json.dumps({'analysis_file':r['file'],'analysis_exit_code':child.returncode}),flush=True)
            except Exception as exc:
                failures.append({'entry': futures[future], 'error': str(exc)})
                print('FAILED', failures[-1], flush=True)
    (a.root/'download_failures.json').write_text(json.dumps(failures, indent=2))
    if failures:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
