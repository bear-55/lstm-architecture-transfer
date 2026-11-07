"""Frozen external replication; public author data read with a class allowlist."""
from pathlib import Path
from zipfile import ZipFile
import io,json,hashlib
import numpy as np
import pandas as pd
from pandas.compat.pickle_compat import Unpickler
from weakness_stats import ROOT,OUT,analyze

ALLOWED={'builtins slice','numpy dtype','numpy ndarray','numpy.core.multiarray _reconstruct',
 'pandas.core.frame DataFrame','pandas.core.indexes.base Index','pandas.core.indexes.base _new_Index',
 'pandas.core.indexes.datetimes DatetimeIndex','pandas.core.indexes.datetimes _new_DatetimeIndex',
 'pandas.core.internals BlockManager','pandas.tseries.offsets Day'}
class Reader(Unpickler):
    def find_class(self,module,name):
        if module+' '+name not in ALLOWED:raise ValueError('Unapproved pickle class: '+module+'.'+name)
        return super().find_class(module,name)

def main():
    freeze=json.loads((OUT/'external_protocol_freeze.json').read_text())
    assert hashlib.sha256((ROOT/freeze['protocol'].replace('\\','/')).read_bytes()).hexdigest()==freeze['english_translation_sha256']
    archive=OUT/'external/Kratzert_et_al_2019_HESSD_LSTM_4_REGIONAL_MODELING.zip'
    inventory=json.loads((OUT/'external/configs_inventory.json').read_text())
    metadata=dict(protocol_sha256=freeze['sha256'],test_period=['1989-10-01','1999-09-30'],
                  training_period=['1999-10-01','2008-09-30'],selection_period=['1989-10-01','1994-09-30'],evaluation_period=['1994-10-01','1999-09-30'],
                  interpretation='Retrospective out-of-calibration temporal validation; these test years precede training years.',
                  hyperparameters='Original paper Appendix B tuned on calibration-period basin cross-validation.',
                  pairing='Original matching seed IDs; no claim of identical initialization or minibatch stream between architectures.',
                  models='B0=concat_static; EA=static input gate; dynamic=no_static',source='https://doi.org/10.4211/hs.83ea5312635e44dc824eeb99eda12f06',license='CC BY 4.0')
    (OUT/'external/eligibility.json').write_text(json.dumps(metadata,indent=2))
    with ZipFile(archive) as z:
        for mse in [False,True]:
            models=['B0','EA','LSTM_dynamic'];seeds=sorted({c['seed'] for c in inventory});all_arrays=[];ref=None;records=[]
            for model in models:
                sims=[]
                for seed in seeds:
                    cs=[c for c in inventory if c['seed']==seed and c['use_mse']==mse and
                        (('LSTM_dynamic' if c['no_static'] else 'B0' if c['concat_static'] else 'EA')==model)]
                    assert len(cs)==1;c=cs[0]
                    folder=str(Path(c['member_path']).parent).replace('\\','/')+'/'
                    files=[n for n in z.namelist() if n.startswith(folder) and n.endswith('.p')];assert len(files)==1
                    raw=z.read(files[0]);obj=Reader(io.BytesIO(raw)).load();basins=sorted(obj)
                    dates=obj[basins[0]].index.to_numpy().astype('datetime64[D]')
                    obs=np.stack([obj[b].qobs.to_numpy() for b in basins]);sim=np.stack([obj[b].qsim.to_numpy() for b in basins])
                    if ref is None:ref=(basins,dates,obs)
                    else:
                        assert basins==ref[0];np.testing.assert_array_equal(dates,ref[1]);np.testing.assert_array_equal(obs,ref[2])
                    for b in basins:np.testing.assert_array_equal(obj[b].index.to_numpy().astype('datetime64[D]'),dates)
                    sims.append(sim);records.append(dict(model=model,seed=seed,pickle_member=files[0],sha256=hashlib.sha256(raw).hexdigest(),config=c))
                all_arrays.append(sims)
            pred=np.asarray(all_arrays);basins,dates,obs=ref
            data={}
            for era,mask in [('earlier',dates<np.datetime64('1994-10-01')),('later',dates>=np.datetime64('1994-10-01'))]:
                assert mask.sum()>=730
                data[era]=[(obs[b,mask],pred[:,:,b,mask]) for b in range(len(basins))]
            name='external_MSE' if mse else 'external_NSE'
            dest=OUT/name;dest.mkdir(parents=True,exist_ok=True)
            (dest/'source_runs.json').write_text(json.dumps(records,indent=2))
            print('EXTERNAL_PREPARED',name,len(basins),len(seeds),pred.shape,flush=True)
            analyze(dict(name=name,models=models,seeds=seeds,basins=basins,series=data))

if __name__=='__main__':main()
