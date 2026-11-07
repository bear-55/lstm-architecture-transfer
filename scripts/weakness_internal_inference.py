"""W2a: inference of frozen models in 370 training basins, outside training years."""
from pathlib import Path
import argparse,gc,hashlib,json,sys,time
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from training import make_model,setup_device
from ea_model import make_ea
from graph_backend import GraphBackend
from final_evaluate import EvaluationWindows,parse_streamflow

OUT=ROOT/'runs/internal_temporal_extension_20261102'
SEEDS=[1009,1021,1033,1051,1063,1087,1097,1109,1123,1151]
PERIODS={'earlier':['1999-10-01','2003-09-30'],'later':['2003-10-01','2008-09-30']}

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load_net(name,seed,cfg,meta,device):
    folder='ea_extension_20261031' if name=='EA' else 'retraining_confirmation_20261029'
    path=ROOT/'runs'/folder/name/f'seed_{seed}'/'best.pt'
    chk=torch.load(path,map_location='cpu',weights_only=True)
    net=make_ea(cfg,meta,seed,device) if name=='EA' else make_model(cfg,meta,name,seed,device)
    net.load_state_dict(chk['model'],strict=True);net.eval()
    if name=='B0':net.forward=lambda d,s,l=None,net=net:net.fused_baseline(d,s)
    return net,path

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['benchmark','run']);parser.add_argument('--batch',type=int,default=1024)
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((ROOT/'configs/formal.json').read_text());meta=json.loads((ROOT/'data/formal/metadata.json').read_text())
    with np.load(ROOT/'data/formal/formal.npz',allow_pickle=False) as z:cache={k:z[k] for k in z.files}
    device=setup_device('cuda')
    dynamic=torch.tensor(cache['x_dynamic'],device=device);static=torch.tensor(cache['x_static'],device=device)
    lags=torch.arange(1-meta['sequence_length'],1,device=device);land=meta['land_indices'];size=args.batch
    def batch(rows,ends):
        rows=torch.as_tensor(rows,dtype=torch.long,device=device);ends=torch.as_tensor(ends,dtype=torch.long,device=device)
        s=static[rows];return(dynamic[rows[:,None],ends[:,None]+lags],s,s[:,land])
    # Cross-hardware and changed-batch check on already scored target predictions.
    benchmarks=[]
    for name in ['B0','M1','EA']:
        folder='ea_evaluation_20261101' if name=='EA' else 'confirmation_evaluation_20261030'
        with np.load(ROOT/'runs'/folder/'later'/name/'seed_1009'/'predictions_test.npz',allow_pickle=False) as z:
            ix=np.linspace(0,len(z['dates'])-1,128,dtype=int)
            rows=np.array([cache['basin_ids'].tolist().index(b) for b in z['basin_ids'][ix]])
            ends=np.searchsorted(cache['dates'],z['dates'][ix]);expected=z['q_predicted_mm_day'][ix]
        inp=batch(np.resize(rows,size),np.resize(ends,size));net,path=load_net(name,1009,cfg,meta,device)
        with torch.no_grad():
            runner=GraphBackend(net,[],[inp]);pred=runner.forward(*inp)[:,0]*meta['q_ref_mm_day'];torch.cuda.synchronize()
            got=pred.cpu().numpy()[:128];np.testing.assert_allclose(got,expected,rtol=1e-4,atol=1e-5)
            started=time.perf_counter()
            for _ in range(10):runner.forward(*inp)
            torch.cuda.synchronize();sec=(time.perf_counter()-started)/10
        row=dict(model=name,batch=size,seconds_per_batch=sec,max_reference_error=float(np.max(np.abs(got-expected))),checkpoint_sha256=digest(path))
        print('BENCHMARK',json.dumps(row),flush=True);benchmarks.append(row)
        del runner,net,inp;gc.collect();torch.cuda.empty_cache()
    (OUT/'benchmark.json').write_text(json.dumps({'device':torch.cuda.get_device_name(0),'batch':size,'results':benchmarks},indent=2),encoding='utf-8')
    if args.action=='benchmark':return
    # Read original observed flow only for the two additional out-of-training periods.
    ids=meta['splits']['train'];assert len(ids)==370
    sources=pd.read_csv(ROOT/'data/manifests/source_files.csv',dtype={'basin_id':str})
    source={kind:sources[sources.kind==kind].set_index('basin_id').path.to_dict() for kind in ['daymet','streamflow_metadata_only']}
    q=np.full((len(ids),len(cache['dates'])),np.nan,dtype=np.float32)
    for j,b in enumerate(ids):
        with Path(source['daymet'][b]).open(encoding='utf-8-sig') as f:f.readline();f.readline();area=float(f.readline())
        ds,values=parse_streamflow(Path(source['streamflow_metadata_only'][b]).read_text(encoding='utf-8-sig'),b,['1999-10-01','2008-09-30'],area)
        q[j,np.searchsorted(cache['dates'],ds)]=values
    datasets={era:EvaluationWindows(cache,meta,ids,p,q) for era,p in PERIODS.items()}
    print('WINDOWS',json.dumps({k:len(v) for k,v in datasets.items()}),flush=True)
    completed=[]
    for name in ['B0','M1','EA']:
        for seed in SEEDS:
            net,path=load_net(name,seed,cfg,meta,device);checksum=digest(path);runner=None
            for era,ds in datasets.items():
                dest=OUT/era/name/f'seed_{seed}';dest.mkdir(parents=True,exist_ok=True)
                prediction=dest/'predictions_test.npz';record=dest/'result.json'
                if prediction.exists() and record.exists():
                    try:r=json.loads(record.read_text())
                    except (json.JSONDecodeError,UnicodeError):r=None
                    if r is not None:
                        assert r['checkpoint_sha256']==checksum and r['prediction_sha256']==digest(prediction)
                        completed.append(r);print('REUSED',name,seed,era,flush=True);continue
                t0=time.perf_counter();pred=np.empty(len(ds),np.float32)
                with torch.no_grad():
                    for start in range(0,len(ds),size):
                        stop=min(start+size,len(ds));rr=ds.rows[start:stop];ee=ds.ends[start:stop]
                        if len(rr)<size:rr=np.resize(rr,size);ee=np.resize(ee,size)
                        inp=batch(rr,ee)
                        if runner is None:runner=GraphBackend(net,[],[inp])
                        result=(runner.forward(*inp)[:,0]*meta['q_ref_mm_day']).cpu().numpy()[:stop-start]
                        if not np.isfinite(result).all():raise RuntimeError('Nonfinite predictions')
                        pred[start:stop]=result
                        if start and start%(size*250)==0:print('PROGRESS',name,seed,era,stop,len(ds),round(time.perf_counter()-t0,1),flush=True)
                data=ds.records(pred)
                with prediction.with_suffix('.tmp').open('wb') as f:np.savez_compressed(f,**data)
                prediction.with_suffix('.tmp').replace(prediction)
                r=dict(model=name,seed=seed,period=era,dates=PERIODS[era],basins=len(ids),windows=len(ds),seconds=time.perf_counter()-t0,
                       checkpoint_sha256=checksum,prediction_sha256=digest(prediction),device=torch.cuda.get_device_name(0),batch=size,training=False)
                record.with_suffix('.tmp').write_text(json.dumps(r,indent=2),encoding='utf-8');record.with_suffix('.tmp').replace(record)
                completed.append(r);print('COMPLETED',name,seed,era,round(r['seconds'],1),flush=True)
            del net,runner;gc.collect();torch.cuda.empty_cache()
    (OUT/'completed.json').write_text(json.dumps(dict(status='completed',runs=30,evaluations=60,basins=370,training=False,results=completed),indent=2),encoding='utf-8')
    print('ALL_60_INTERNAL_EVALUATIONS_COMPLETED',flush=True)

if __name__=='__main__':main()
