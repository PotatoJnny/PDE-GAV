#!/usr/bin/env python3
"""Freeze only the source and configs needed by the first approved NS batch."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
EXTENSIONS={'.py','.json','.toml','.md','.sh','.txt','.yaml','.yml','.csv','.f90','.f','.h','.c'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-root',required=True,type=Path)
    args=parser.parse_args()
    batch=args.batch_root.resolve()
    if batch.exists(): raise FileExistsError(batch)
    snapshot=batch/'source'; snapshot.mkdir(parents=True)
    folders=('src','scripts','legacy/ns/src','configs/paper/ns_draft_v1','configs/paper/ns_gp_v2','configs/paper/ns_bgp_v3')
    files=[ROOT/'pyproject.toml',ROOT/'README.md']
    for folder in folders:
        files.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and
                     (p.suffix.lower() in EXTENSIONS or p.name.startswith(('LICENSE','NOTICE'))) and '__pycache__' not in p.parts)
    files.extend((ROOT/'docs/paper/evidence').glob('ns_draft_v1*'))
    files=[p for p in files if p.is_file()]
    records=[]
    for source in sorted(set(files)):
        relative=source.relative_to(ROOT)
        target=snapshot/relative; target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
        records.append({'path':str(relative),'sha256':sha256(target.read_bytes()).hexdigest(),'bytes':target.stat().st_size})
    git=['git',f'--git-dir={ROOT/".git-real"}',f'--work-tree={ROOT}']
    commit=subprocess.run([*git,'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
    status=subprocess.run([*git,'status','--short'],capture_output=True,text=True,check=True).stdout
    manifest={'schema':'ns_draft_source_snapshot_v1','source_root':str(ROOT),'source_base_commit':commit,
              'source_status':status,'created_at_unix':time.time(),'files':records,
              'scope':'Source/config snapshot; original numerical data and frozen Generator checkpoints are external and separately pinned.'}
    (snapshot/'snapshot_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for directory in ('control','logs','runs','job_status','preflight'): (batch/directory).mkdir()
    print(json.dumps({'batch_root':str(batch),'source_files':len(records),'source_bytes':sum(r['bytes'] for r in records),
                      'snapshot_manifest_sha256':sha256((snapshot/'snapshot_manifest.json').read_bytes()).hexdigest()}))


if __name__=='__main__':main()
