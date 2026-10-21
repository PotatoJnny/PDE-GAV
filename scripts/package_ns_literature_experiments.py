#!/usr/bin/env python3
"""Freeze the four PDE-Agent methods as a self-contained source/runbook archive.

Only source, templates, tests and this experiment's instructions are included.
Model authentication, running experiments and numerical assets are not copied.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[1]
GUIDE = Path('docs/operations/NS_CODEPDE_AUTONUMERICS_20260920')


def build(output, *, make_archive=True):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    archive = output.with_suffix('.tar.gz')
    if archive.exists():
        raise FileExistsError(archive)
    folders = ('src', 'legacy/ns/src', 'scripts', 'tests',
               'configs/paper/ns_literature_v1', 'configs/paper/ns_literature_v2', str(GUIDE))
    files = [ROOT / 'pyproject.toml', ROOT / 'README.md']
    for folder in folders:
        files.extend(p for p in (ROOT / folder).rglob('*') if p.is_file()
                     and '__pycache__' not in p.parts and not p.name.endswith(('.pyc', '.orig')))
    output.mkdir(parents=True)
    for source in sorted(set(files)):
        if source.is_symlink():
            raise ValueError('release sources must be regular files: ' + str(source))
        relative = source.relative_to(ROOT)
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        target.chmod(0o755 if source.suffix == '.sh' else 0o644)
    shutil.copyfile(output / 'README.md', output / 'PROJECT_README.md')
    shutil.copyfile(output / GUIDE / 'RUN_PACKAGE.md', output / 'README.md')
    shutil.copyfile(output / GUIDE / 'BINDING.example.json', output / 'BINDING.example.json')
    shutil.copyfile(output / GUIDE / 'PROTOCOL.json', output / 'EXPERIMENT_PROTOCOL.json')
    shutil.copyfile(output / GUIDE / 'requirements-ns-experiment.txt', output / 'requirements-ns-experiment.txt')
    entries = {str(p.relative_to(output)): sha256(p.read_bytes()).hexdigest()
               for p in sorted(output.rglob('*')) if p.is_file()}
    manifest = {'schema':'ns_four_pde_agents_source_package_v1', 'scientific_protocol':'completion_v2',
                'files':entries, 'numerical_assets':'bind original v6 ladder assets on receiver',
                'authentication':'receiver-local only; not included'}
    (output / 'PACKAGE_MANIFEST.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
    (output / 'SHA256SUMS').write_text(''.join(f'{h}  {p}\n' for p,h in entries.items()))
    if not make_archive:
        return {'directory':str(output), 'files':len(entries)}
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(output, arcname=output.name)
    digest = sha256(archive.read_bytes()).hexdigest()
    archive.with_name(archive.name+'.sha256').write_text(f'{digest}  {archive.name}\n')
    return {'directory':str(output), 'archive':str(archive), 'sha256':digest,
            'files':len(entries), 'bytes':archive.stat().st_size}

def build_ready(output, binding_path):
    """Package source and already exported data; receiver binds local authentication."""
    output = Path(output).resolve()
    binding_path = Path(binding_path).resolve()
    binding = json.loads(binding_path.read_text())
    if binding.get('protocol') != 'completion_v2' or binding.get('methods') != [
            'codepde', 'autonumerics', 'autopde', 'pde_sharp']:
        raise ValueError('ready package requires the agreed four-method completion binding')
    if output.exists() or output.with_suffix('.tar.gz').exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    source_result = build(output / 'code', make_archive=False)
    data = output / 'data'
    data.mkdir(mode=0o700)
    def original(value):
        p = Path(value).expanduser()
        return p.resolve() if p.is_absolute() else (binding_path.parent / p).resolve()
    def copy_file(src, dst):
        if src.is_symlink() or not src.is_file():
            raise ValueError('missing or nonregular data file: '+str(src))
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        dst.chmod(0o600)
    for key in ('starter_manifest', 'selection_manifest', 'terminal_manifest'):
        src = original(binding[key])
        copy_file(src, data / src.name)
        assets = src.with_suffix('.cases')
        if not assets.is_dir():
            raise FileNotFoundError(assets)
        for item in assets.rglob('*'):
            if item.is_symlink(): raise ValueError('data links are not portable: '+str(item))
            if item.is_file(): copy_file(item, data / assets.name / item.relative_to(assets))
        binding[key] = 'data/' + src.name
    for key in ('flow_checkpoint', 'pca_checkpoint'):
        src = original(binding['generator'][key])
        target = data / 'generator_v6' / src.name
        copy_file(src, target)
        binding['generator'][key] = str(target.relative_to(output))
    # The data definition accompanies the actual weights, without exporting the old runtime.
    raw_binding = json.loads(binding_path.read_text())
    definition = original(raw_binding['generator']['flow_checkpoint']).parent / 'generator_definition.json'
    if definition.is_file(): copy_file(definition, data / 'generator_v6' / definition.name)
    initial = original(binding['initial_bundle'])
    for item in initial.parent.rglob('*'):
        if item.is_symlink(): raise ValueError('S0 links are not portable')
        if item.is_file(): copy_file(item, data / 's0' / item.relative_to(initial.parent))
    binding['initial_bundle'] = 'data/s0/' + initial.name
    binding['codex_binary'] = None
    binding['auth_file'] = None
    (output / 'BINDING.template.json').write_text(json.dumps(binding, indent=2)+'\n')
    (output / 'prepare.py').write_text('''#!/usr/bin/env python3
"""Bind this received package to local paths; never submit a job or call a model."""
import argparse, json, shutil, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT/'code/src'), str(ROOT/'code/legacy/ns/src')]
from pde_agent_verifier_solver.plugins.ns.baselines.preparation import prepare_comparisons
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--codex-binary', type=Path)
p.add_argument('--auth-file', type=Path, default=Path.home()/'.codex/auth.json')
a = p.parse_args()
binary = a.codex_binary or shutil.which('codex')
if not binary: p.error('set --codex-binary to the installed Agent executable')
output = a.output.expanduser().resolve()
if output.exists() and any(output.iterdir()): p.error('use a new or empty output directory')
b = json.loads((ROOT/'BINDING.template.json').read_text())
for key in ('starter_manifest','selection_manifest','terminal_manifest','initial_bundle'):
    b[key] = str((ROOT/b[key]).resolve())
for key in ('flow_checkpoint','pca_checkpoint'):
    b['generator'][key] = str((ROOT/b['generator'][key]).resolve())
b['codex_binary'] = str(Path(binary).expanduser().resolve())
b['auth_file'] = str(a.auth_file.expanduser().resolve())
output.mkdir(parents=True,exist_ok=True)
binding = output/'receiver_binding.json'
binding.write_text(json.dumps(b,indent=2)+'\\n')
binding.chmod(0o600)
print(prepare_comparisons(binding,output/'experiment'))
''')
    (output / 'README.md').write_text('''# Four PDE-Agent methods x fixed / online environments: complete run package

The package contains the source, the original v6 development set of 64 cases, the 24 selection cases, the 96 terminal cases, the actual weights and S0. The development arrays were exported from the same saved continuous Cases; no cases were resampled and no training was repeated. No access to the original account directories is required.

1. Unpack this package in the target account and use an existing scientific Python environment.
2. Run `python prepare.py --output /absolute/path/to/new_experiment`. By default it uses `codex` from PATH and `~/.codex/auth.json`; `--codex-binary` and `--auth-file` override them. The package contains no authentication files.
3. That command only binds paths, checks the data and writes the eight configurations; it does not start a model or submit jobs.
4. After obtaining one GPU allocation, run from the package directory:

```
python run.py --experiment /absolute/path/to/new_experiment --group codepde_fixed_sampled
```

The eight groups are `codepde`, `autonumerics`, `autopde` and `pde_sharp`, each followed by `_fixed_sampled` or `_continuous_online_base`. Submit each group independently; each segment lasts three hours, and after exit code 75 the same command with `--resume` continues. One GPU with 16 CPUs and 64 GB, four parallel candidates, and 300 seconds of solving per case. The Agent phase ends naturally; there is no research-time or cumulative case limit.

The online Generator keeps the original v6 weights, raw Flow and 256 Heun steps. Fixed and online share the same finite inputs and references of the 64 cases; online additionally exposes continuous inputs and fresh Base cases. Selection uses 96 hidden points per case on 24 cases and the terminal evaluation uses 128 hidden points on 96 cases, both with equal Base, Guided and Base-perturbed shares.

The Slurm template, the stages of the four methods, the terminal evaluation, result files and metric definitions are described in the run package documentation shipped under `code/docs/operations/`. Source is under `code/`.

This package passed the CPU data-binding check. Real model calls, CUDA and end-to-end timing on the target compute node must be confirmed at the formal launch. No experiment job was submitted by this packaging step.
''')
    entries = {str(p.relative_to(output)): sha256(p.read_bytes()).hexdigest()
               for p in sorted(output.rglob('*')) if p.is_file()}
    (output / 'CONTENTS.json').write_text(json.dumps({'schema':'ns_four_agents_ready_package_v1',
        'files':entries,'authentication_included':False,'development_cases':64,
        'selection_cases':24,'terminal_cases':96},indent=2)+'\n')
    archive = output.with_suffix('.tar.gz')
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(output, arcname=output.name)
    return {'directory':str(output),'archive':str(archive),'files':len(entries),
            'bytes':archive.stat().st_size,'source_files':source_result['files']}



if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--binding',type=Path,help='include already exported v6 assets from this binding')
    args=parser.parse_args()
    result=build_ready(args.output,args.binding) if args.binding else build(args.output)
    print(json.dumps(result,indent=2))
