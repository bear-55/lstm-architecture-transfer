"""Fixed EA checkpoints: validation reproduction, then two-period Test inference.

No training or configuration selection. Existing B0 predictions are verified
and reused, with exactly the same observations and dates.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
import ea_extension as ea
import retraining_confirmation as cohort
import confirmation_evaluate as shared
import final_evaluate as base
from ea_model import make_ea
from evaluate import summarize
from project import ROOT,digest
from training import setup_device

OUT=ROOT/'runs/ea_evaluation_20261101'
BUNDLE=ROOT/'ea_evaluation_bundle_20261101.json'
PERIODS=('earlier','later')


def read_npz(path):
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k] for k in z.files}


def prepare(device=None):
    manifest=base.read_json(BUNDLE)
    for n,h in {**manifest['new_files'],**manifest['shared_files']}.items():
        p=(ROOT/n).resolve()
        base.require(p.is_relative_to(ROOT) and digest(p)==h,'Evaluation source changed: '+n)
    cfg,meta,spec=ea.load()
    seal=base.read_json(ea.OUT/'cohort_completed.json')
    base.require(seal['status']=='completed' and seal['runs']==10 and seal['seeds']==spec['formal_seeds']
        and not seal['test_scored'] and seal['paired_B0_sampling_checked'] and seal['provenance']==ea.provenance(),
        'Complete frozen EA training batch required')
    expected={f'EA/seed_{s}/{n}' for s in spec['formal_seeds'] for n in cohort.REQUIRED+('completed_artifacts.json',)}
    base.require(set(seal['files'])==expected,'Unexpected EA training artifacts')
    base.verify_artifacts(ea.OUT,seal['files'])
    for seed in spec['formal_seeds']:
        base.require(ea.verify(seed),'Missing completed EA checkpoint')
        cohort.validate_completed(ea.run_path(seed),ea.identity(seed))
    original=shared.verify_bundle()
    protocol=base.read_json(ROOT/'protocol/ea_extension_20261031.json')
    base.require(manifest['periods']==original['periods'] and
        manifest['periods']['earlier']==protocol['selection_period'] and
        manifest['periods']['later']==protocol['evaluation_period'],'Period differs from fixed protocol')
    prior=base.read_json(shared.OUT/'completed.json')
    context=base.read_json(shared.OUT/'evaluation_context.json')
    base.require(prior['status']=='completed' and prior['evaluations']==40 and prior['identity']==context['identity'],
        'Completed original B0/M1 inference required')
    # Do not recompute baseline predictions; verify every B0 file against its seal.
    for period in PERIODS:
        for seed in spec['formal_seeds']:
            prefix=f'{period}/B0/seed_{seed}/'
            for name in ('result.json','metrics_per_basin.csv','predictions_test.npz'):
                path=shared.OUT/(prefix+name)
                base.require(prior['artifacts'].get(prefix+name)==digest(path),'Baseline prediction changed')
            result=base.read_json(shared.OUT/prefix/'result.json')
            base.require(result['identity']==prior['identity'] and result['model']=='B0' and result['seed']==seed
                and result['period']==period and result['status']=='completed'
                and result['checkpoint_sha256']==digest(cohort.OUT/'B0'/f'seed_{seed}'/'best.pt'),
                'Baseline checkpoint/prediction provenance mismatch')
    if device is None:
        return cfg,meta,spec,manifest
    cohort.check_versions()
    context=dict(bundle_sha256=digest(BUNDLE),environment=cohort.runtime(device),
        cohort_sha256=digest(ea.OUT/'cohort_completed.json'),baseline_completion_sha256=digest(shared.OUT/'completed.json'))
    identity=hashlib.sha256(json.dumps(context,sort_keys=True).encode()).hexdigest()
    p=OUT/'evaluation_context.json'
    if p.exists():
        base.require(base.read_json(p)==dict(identity=identity,context=context),'Inference context changed; preserve outputs')
    else:
        base.save_json(p,dict(identity=identity,context=context))
    cache=read_npz(ROOT/'data/formal/formal.npz')
    return cfg,meta,spec,manifest,cache,identity


def load_network(cfg,meta,seed,device):
    p=ea.run_path(seed)
    result=base.read_json(p/'result.json')
    best=torch.load(p/'best.pt',map_location='cpu',weights_only=True)
    base.require(best['identity']==result['identity']==ea.identity(seed) and best['epoch']==result['best_epoch'],
        'Validation-selected checkpoint identity differs')
    net=make_ea(cfg,meta,seed,device)
    net.load_state_dict(best['model'],strict=True)
    base.require(all(torch.isfinite(v).all().item() for v in net.state_dict().values()),'Nonfinite weights')
    net.eval()
    return net,result,digest(p/'best.pt')


def cleanup(net,device):
    shared.release(net)
    gc.collect()
    if device.type=='cuda': torch.cuda.empty_cache()


def check_validation(cfg,meta,spec,cache,device,identity):
    val=base.EvaluationWindows(cache,meta,meta['splits']['val'],cfg['periods']['development_labels'],cache['q_val'])
    checks=[]
    for seed in spec['formal_seeds']:
        path=OUT/'validation_checks'/f'seed_{seed}.json'
        checksum=digest(ea.run_path(seed)/'best.pt')
        if path.exists():
            r=base.read_json(path)
            base.require(r['status']=='passed' and r['identity']==identity and r['seed']==seed and
                r['checkpoint_sha256']==checksum and r['prediction_count']==len(val),'Validation resume mismatch')
        else:
            started=time.perf_counter()
            net,source,h=load_network(cfg,meta,seed,device)
            pred=base.predict(net,val,cfg,meta,device,f'Validation EA/{seed}')
            r=base.compare_validation(val.records(pred),read_npz(ea.run_path(seed)/'predictions_val.npz'),
                base.metric_rows(val,pred,cfg),source['validation'])
            r.update(status='passed',identity=identity,seed=seed,model='EA',checkpoint_sha256=h,
                prediction_count=len(val),seconds=time.perf_counter()-started)
            base.save_json(path,r)
            cleanup(net,device); del net
        checks.append(r)
        print(f'Validation verified EA/{seed}',flush=True)
    base.save_json(OUT/'validation_recheck.json',dict(status='passed',identity=identity,runs=checks))
    return checks


def require_checks(checks,seeds,identity):
    base.require(len(checks)==len(seeds) and {r['seed'] for r in checks}==set(seeds)
        and all(r['status']=='passed' and r['identity']==identity for r in checks),'All ten validation checks required')


def evaluate(cfg,meta,spec,manifest,cache,device,identity,checks):
    require_checks(checks,spec['formal_seeds'],identity)
    done=OUT/'completed.json'
    if done.exists():
        r=base.read_json(done)
        base.require(r['status']=='completed' and r['identity']==identity and r['evaluations']==20,'Completion differs')
        base.verify_artifacts(OUT,r['artifacts'])
        print('ALL_20_EA_PERIOD_PREDICTIONS_COMPLETED_AND_VERIFIED',flush=True)
        return
    results=[]
    for period in PERIODS:
        obs=read_npz(shared.DATA/f'{period}_observations.npz')
        ds=shared.dataset_from_records(cache,meta,manifest['periods'][period],obs)
        base.require(len(ds)==manifest['observations'][period]['count'],'Observation count differs')
        for seed in spec['formal_seeds']:
            # Exact B0 observation/date alignment is enforced for every comparison.
            shared.check_prediction(ds,read_npz(shared.OUT/period/'B0'/f'seed_{seed}'/'predictions_test.npz'))
            directory=OUT/period/'EA'/f'seed_{seed}'
            directory.mkdir(parents=True,exist_ok=True)
            path=directory/'result.json'
            checksum=digest(ea.run_path(seed)/'best.pt')
            if path.exists():
                r=base.read_json(path)
                base.require(r['identity']==identity and r['status']=='completed' and r['model']=='EA' and
                    r['seed']==seed and r['period']==period and r['checkpoint_sha256']==checksum,'Result resume mismatch')
                base.verify_artifacts(directory,r['artifacts'])
                pred=shared.check_prediction(ds,read_npz(directory/'predictions_test.npz'))
                base.require(summarize(base.metric_rows(ds,pred,cfg))==r['summary'],'Resumed metrics differ')
            else:
                started=time.perf_counter()
                net,source,h=load_network(cfg,meta,seed,device)
                base.require(h==checksum,'Checkpoint changed')
                pred=base.predict(net,ds,cfg,meta,device,f'{period} EA/{seed}')
                rows=base.metric_rows(ds,pred,cfg)
                base.save_predictions(directory/'predictions_test.npz',ds.records(pred))
                base.save_csv(directory/'metrics_per_basin.csv',rows)
                r=dict(status='completed',identity=identity,model='EA',seed=seed,period=period,
                    checkpoint_sha256=h,best_epoch=source['best_epoch'],prediction_count=len(ds),
                    seconds=time.perf_counter()-started,summary=summarize(rows),
                    artifacts={n:digest(directory/n) for n in ('predictions_test.npz','metrics_per_basin.csv')})
                base.save_json(path,r)
                cleanup(net,device); del net
            results.append(dict(period=period,seed=seed,model='EA',best_epoch=r['best_epoch'],**r['summary']))
            print(f'Prediction complete {period} EA/{seed} ({len(results)}/20)',flush=True)
    base.save_csv(OUT/'run_summary.csv',results)
    artifacts={p.relative_to(OUT).as_posix():digest(p) for p in OUT.rglob('*') if p.is_file()
        and p.suffix in ('.json','.csv','.npz') and p!=done}
    base.save_json(done,dict(status='completed',identity=identity,evaluations=20,basins=107,
        B0_predictions_reused=True,artifacts=artifacts))
    print('ALL_20_EA_PERIOD_PREDICTIONS_COMPLETED_AND_VERIFIED',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['plan','check','run'])
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    a=p.parse_args()
    if a.action=='plan':
        cfg,meta,spec,manifest=prepare()
        print(json.dumps(dict(training=False,models=['EA'],seeds=spec['formal_seeds'],validation_rechecks=10,
            period_evaluations=20,test_basins=len(meta['splits']['test']),periods=manifest['periods'],
            B0_predictions_reused=True,test_scored=False,output=str(OUT)),indent=2))
        return
    if a.device=='cuda':
        active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
        base.require(not active.strip(),'Another GPU compute process is active')
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'evaluation.lock').open('a+b') as lock:
        if os.name=='nt':
            import msvcrt
            if lock.tell()==0: lock.write(b'0'); lock.flush()
            lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        device=setup_device(a.device)
        cfg,meta,spec,manifest,cache,identity=prepare(device)
        checks=check_validation(cfg,meta,spec,cache,device,identity)
        if a.action=='run': evaluate(cfg,meta,spec,manifest,cache,device,identity,checks)
        else: print('ALL_10_EA_VALIDATION_RECHECKS_PASSED; no Test inference',flush=True)


if __name__=='__main__': main()
