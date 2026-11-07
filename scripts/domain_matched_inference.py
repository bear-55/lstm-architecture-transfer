"""A: CPU-only fixed-checkpoint inference for the aligned 107-basin period."""
from pathlib import Path
import argparse,gc,json,sys,time,hashlib
import numpy as np,pandas as pd,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from training import setup_device
from final_evaluate import EvaluationWindows,parse_streamflow
from weakness_internal_inference import load_net,SEEDS,digest
OUT=ROOT/'runs/domain_matched_20261102'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['benchmark','run']);ap.add_argument('--batch',type=int,default=256)
    ap.add_argument('--shard',type=int,default=0);ap.add_argument('--shards',type=int,default=1);args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((ROOT/'configs/formal.json').read_text());meta=json.loads((ROOT/'data/formal/metadata.json').read_text())
    with np.load(ROOT/'data/formal/formal.npz',allow_pickle=False) as z:cache={k:z[k] for k in z.files}
    device=setup_device('cpu');size=args.batch;dynamic=torch.tensor(cache['x_dynamic']);static=torch.tensor(cache['x_static']);lags=torch.arange(1-meta['sequence_length'],1)
    def batch(rows,ends):
        rows=torch.as_tensor(rows,dtype=torch.long);ends=torch.as_tensor(ends,dtype=torch.long);s=static[rows]
        return(dynamic[rows[:,None],ends[:,None]+lags],s,s[:,meta['land_indices']])
    checks=[]
    for name in ['B0','M1','EA']:
        folder='ea_evaluation_20261101' if name=='EA' else 'confirmation_evaluation_20261030'
        with np.load(ROOT/'runs'/folder/'later'/name/'seed_1009'/'predictions_test.npz',allow_pickle=False) as z:
            ix=np.linspace(0,len(z['dates'])-1,128,dtype=int);rows=np.array([cache['basin_ids'].tolist().index(b) for b in z['basin_ids'][ix]])
            ends=np.searchsorted(cache['dates'],z['dates'][ix]);expected=z['q_predicted_mm_day'][ix]
        net,path=load_net(name,1009,cfg,meta,device);inp=batch(np.resize(rows,size),np.resize(ends,size))
        with torch.no_grad():
            t=time.perf_counter();got=(net(*inp)[:,0]*meta['q_ref_mm_day']).numpy()[:128];sec=time.perf_counter()-t
        np.testing.assert_allclose(got,expected,rtol=1e-4,atol=1e-5)
        checks.append(dict(model=name,batch=size,seconds=sec,max_abs_error=float(np.max(np.abs(got-expected)))))
        print('CPU_BENCHMARK',checks[-1],flush=True);del net,inp;gc.collect()
    (OUT/f'benchmark_shard{args.shard}.json').write_text(json.dumps(checks,indent=2))
    if args.action=='benchmark':return
    ids=meta['splits']['test'];assert len(ids)==107;period=['1999-10-01','2003-09-30']
    sf=pd.read_csv(ROOT/'data/manifests/source_files.csv',dtype={'basin_id':str})
    source={k:sf[sf.kind==k].set_index('basin_id').path.to_dict() for k in ['daymet','streamflow_metadata_only']}
    q=np.full((len(ids),len(cache['dates'])),np.nan,dtype=np.float32)
    for j,b in enumerate(ids):
        with Path(source['daymet'][b]).open(encoding='utf-8-sig') as f:f.readline();f.readline();area=float(f.readline())
        ds,values=parse_streamflow(Path(source['streamflow_metadata_only'][b]).read_text(encoding='utf-8-sig'),b,period,area)
        q[j,np.searchsorted(cache['dates'],ds)]=values
    ds=EvaluationWindows(cache,meta,ids,period,q);completed=[];print('WINDOWS',len(ds),flush=True)
    tasks=[(m,s) for m in ['B0','M1','EA'] for s in SEEDS]
    chosen=set(tasks[args.shard::args.shards])
    for name in ['B0','M1','EA']:
        for seed in SEEDS:
            if (name,seed) not in chosen:continue
            net,path=load_net(name,seed,cfg,meta,device);dest=OUT/'earlier'/name/f'seed_{seed}';dest.mkdir(parents=True,exist_ok=True)
            predfile=dest/'predictions_test.npz';record=dest/'result.json';ck=digest(path)
            if predfile.exists() and record.exists():
                r=json.loads(record.read_text());assert r['checkpoint_sha256']==ck and r['prediction_sha256']==digest(predfile)
                completed.append(r);print('REUSED',name,seed,flush=True);continue
            t=time.perf_counter();pred=np.empty(len(ds),np.float32)
            with torch.no_grad():
                for start in range(0,len(ds),size):
                    stop=min(start+size,len(ds));out=net(*batch(ds.rows[start:stop],ds.ends[start:stop]))
                    pred[start:stop]=(out[:,0]*meta['q_ref_mm_day']).numpy()
            assert np.isfinite(pred).all()
            with predfile.with_suffix('.tmp').open('wb') as f:np.savez_compressed(f,**ds.records(pred))
            predfile.with_suffix('.tmp').replace(predfile)
            r=dict(model=name,seed=seed,period=period,basins=107,windows=len(ds),seconds=time.perf_counter()-t,device='cpu',threads=torch.get_num_threads(),
                   checkpoint_sha256=ck,prediction_sha256=digest(predfile),training=False)
            record.with_suffix('.tmp').write_text(json.dumps(r,indent=2));record.with_suffix('.tmp').replace(record)
            completed.append(r);print('COMPLETED',name,seed,round(r['seconds'],1),flush=True);del net;gc.collect()
    (OUT/f'completed_shard{args.shard}.json').write_text(json.dumps(dict(status='completed',evaluations=len(completed),basins=107,training=False,results=completed),indent=2))
    print('SHARD_ALIGNED_PERIOD_EVALUATIONS_COMPLETED',args.shard,len(completed),flush=True)

if __name__=='__main__':main()
