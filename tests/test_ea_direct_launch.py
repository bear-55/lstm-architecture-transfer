import importlib.util
import sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'cloud'))
import run_ea_formal_direct as direct


def test_stop_matches_only_our_development_commands():
    assert direct.development_kind(['python','-u','cloud/run_ea_extension.py','develop'],ROOT)=='development_queue'
    assert direct.development_kind(['python','-u','src/ea_extension.py','--seed','2027','--development'],ROOT)=='development_worker'
    for args,cwd in [(['python','cloud/run_ea_extension.py','run'],ROOT),
                     (['python','src/ea_extension.py','--seed','1009'],ROOT),
                     (['python','src/ea_extension.py','--seed','2027','--development'],ROOT/'unrelated'),
                     (['python','src/ea_extension.py','--seed','1009','--development'],ROOT),
                     (['python','unrelated.py','develop'],ROOT)]:
        assert direct.development_kind(args,cwd) is None


def test_pid_reuse_does_not_signal_unrelated_process(monkeypatch):
    sent=[]
    original=dict(pid=123,starttime='1',kind='development_worker',argv=['old'])
    monkeypatch.setattr(direct,'process_info',lambda pid:dict(original,starttime='2'))
    monkeypatch.setattr(direct.os,'kill',lambda *args:sent.append(args))
    direct.signal_if_unchanged(original)
    assert not sent
    monkeypatch.setattr(direct,'process_info',lambda pid:original)
    direct.signal_if_unchanged(original)
    assert len(sent)==1 and sent[0][0]==123


def test_direct_queue_never_calls_development_summary(monkeypatch):
    # Exercise production queue loop without a GPU, process signaling or file writes.
    calls=[]
    def forbidden(*args):
        raise AssertionError('Must not require or run development again')
    monkeypatch.setattr(direct.queue,'development_summary',forbidden)
    monkeypatch.setattr(direct.queue,'launch',lambda seed,dev:calls.append((seed,dev)))
    monkeypatch.setattr(direct.queue,'collect',lambda spec:calls.append('collected'))
    p=direct.plan()
    direct.run_formal_tasks({'formal_seeds':p['formal_seeds']})
    assert calls==[(s,False) for s in p['formal_seeds']]+['collected']
    assert p['formal_runs']==10 and p['require_completed_development'] is False
    assert 2027 not in p['formal_seeds']
    import ea_extension as ea
    assert p['provenance']==ea.provenance()
