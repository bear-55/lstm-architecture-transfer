"""Additive EA-LSTM development/formal queue; no Test scoring."""
import argparse
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'cloud'))
import ea_extension as ea
import retraining_confirmation as cohort
from run_retraining_confirmation import gpu_environment,require_idle_gpu
from project import save_json,digest

PREFLIGHT=ROOT/'runs/ea_preflight_20261031'
LOGS=ROOT/'runs/ea_logs_20261031'


def check_ready():
    from training import setup_device
    cohort.check_versions()
    r=json.loads((PREFLIGHT/'ready.json').read_text(encoding='utf-8'))
    if r['status']!='passed' or r['provenance']!=ea.provenance() or r['environment']!=cohort.runtime(setup_device('cuda')):
        raise RuntimeError('EA preflight code/data/environment mismatch')


def preflight():
    require_idle_gpu()
    PREFLIGHT.mkdir(parents=True,exist_ok=True)
    (PREFLIGHT/'ready.json').unlink(missing_ok=True)
    log=PREFLIGHT/'preflight.log'
    print('EA preflight started: '+str(log),flush=True)
    with log.open('w',encoding='utf-8') as f:
        subprocess.run([sys.executable,'-u','src/ea_checks.py','--output',str(PREFLIGHT/'ready.json')],
            cwd=ROOT,env=gpu_environment(),stdout=f,stderr=subprocess.STDOUT,check=True)
    print(log.read_text(encoding='utf-8'),flush=True)


def launch(seed,development):
    LOGS.mkdir(parents=True,exist_ok=True)
    log=LOGS/f'{"development" if development else "EA"}_seed_{seed}.log'
    cmd=[sys.executable,'-u','src/ea_extension.py','--seed',str(seed)]
    if development:
        cmd.append('--development')
    child=None
    try:
        with log.open('a',encoding='utf-8') as f:
            print(f'Started EA seed={seed}; development={development}; log={log}',flush=True)
            child=subprocess.Popen(cmd,cwd=ROOT,env=gpu_environment(),stdout=f,stderr=subprocess.STDOUT)
            code=child.wait()
        if code:
            raise RuntimeError(f'EA seed={seed} failed with code {code}; queue stopped; see {log}')
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


def development_summary(spec):
    seed=spec['development_seed']
    if not ea.verify(seed,True):
        raise RuntimeError('Development run is not complete')
    p=ea.run_path(seed,True)
    r=json.loads((p/'result.json').read_text(encoding='utf-8'))
    with (p/'history.csv').open(newline='') as f:
        hist=list(csv.DictReader(f))
    out=dict(status='completed',seed=seed,excluded_from_formal=True,test_scored=False,
        best_epoch=r['best_epoch'],completed_epochs=r['completed_epochs'],validation=r['validation'],
        validation_median_nse_by_epoch=[float(h['validation_median_nse']) for h in hist],
        train_loss_by_epoch=[float(h['train_loss']) for h in hist],
        stopped_by_patience=r['stopped_by_patience'],
        best_in_last_three_epochs=r['best_epoch']>r['completed_epochs']-3,
        note='Inspect validation/convergence before formal queue. Do not revise using Test scores.')
    save_json(ea.OUT/'development_summary.json',out)
    print(json.dumps(out,indent=2),flush=True)
    return out


def collect(spec):
    cfg,_,_=cohort.load()
    hashes=cohort.provenance()
    files={}
    for s in spec['formal_seeds']:
        p=ea.run_path(s)
        if not ea.verify(s):
            raise RuntimeError(f'EA {s} incomplete')
        cohort.validate_completed(p,ea.identity(s))
        bp=cohort.OUT/'B0'/f'seed_{s}'
        key=cohort.identity_for('B0',s,hashes,False,'native_fused')
        if not cohort.check_existing(bp,'B0',s,cfg,hashes,key):
            raise RuntimeError('Existing B0 is not completed')
        with (p/'history.csv').open(newline='') as f:
            eh=list(csv.DictReader(f))
        with (bp/'history.csv').open(newline='') as f:
            bh=list(csv.DictReader(f))
        for a,b in zip(eh,bh):
            for field in ('epoch','sampling_sha256','sampled_windows','optimizer_steps'):
                if a[field]!=b[field]:
                    raise RuntimeError(f'Paired B0/EA sampler mismatch: seed={s}, field={field}')
        for name in cohort.REQUIRED+('completed_artifacts.json',):
            files[(p/name).relative_to(ea.OUT).as_posix()]=digest(p/name)
    save_json(ea.OUT/'cohort_completed.json',dict(status='completed',runs=10,seeds=spec['formal_seeds'],
        files=files,provenance=ea.provenance(),paired_B0_sampling_checked=True,test_scored=False))
    print('ALL_10_EA_RUNS_COMPLETED_AND_VERIFIED. Test evaluation remains separate.',flush=True)


def status(spec):
    for dev,s in [(True,spec['development_seed'])]+[(False,s) for s in spec['formal_seeds']]:
        p=ea.run_path(s,dev)
        state='completed' if (p/'result.json').exists() else 'checkpoint exists' if (p/'last.pt').exists() else 'pending'
        print(f'EA seed={s} development={dev}: {state}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['plan','preflight','develop','development-summary','run','status','collect'])
    a=p.parse_args()
    cfg,meta,spec=ea.load()
    if a.action=='plan':
        from ea_model import make_ea
        import torch
        net=make_ea(cfg,meta,spec['development_seed'],torch.device('cpu'))
        print(json.dumps(dict(model='EA',formal_seeds=spec['formal_seeds'],development_seed=spec['development_seed'],
            development_excluded=True,formal_runs=10,parameters=sum(p.numel() for p in net.parameters()),
            dynamic_features=len(meta['dynamic_names']),static_features=len(meta['static_names']),
            batch_size=cfg['training']['batch_size'],max_epochs=cfg['training']['max_epochs'],
            test_labels_used=False,output=str(ea.OUT)),indent=2))
        return
    if a.action=='status':
        status(spec)
        return
    if a.action=='development-summary':
        development_summary(spec)
        return
    if os.name!='posix':
        raise RuntimeError('Launch GPU queue on Linux cloud host')
    import fcntl
    def stop(signum,frame):
        raise KeyboardInterrupt(f'Received {signum}')
    signal.signal(signal.SIGTERM,stop)
    lock=ROOT/'runs/confirmation_queue.lock'  # Share training lock with B0/M1.
    lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if a.action=='preflight':
            preflight()
        elif a.action=='collect':
            collect(spec)
        else:
            require_idle_gpu()
            check_ready()
            if a.action=='develop':
                launch(spec['development_seed'],True)
                development_summary(spec)
                print('EA_DEVELOPMENT_COMPLETED. Formal ten-seed queue has not started.',flush=True)
            else:
                development_summary(spec)
                for seed in spec['formal_seeds']:
                    launch(seed,False)
                    print(f'Completed EA seed={seed}',flush=True)
                collect(spec)


if __name__=='__main__':
    main()
