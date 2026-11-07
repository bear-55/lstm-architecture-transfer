"""Stop the excluded development task and run the frozen formal ten-seed EA queue.

This additive entry leaves the already validated training implementation and
preflight provenance unchanged. The dated workflow amendment is recorded.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'cloud'))
import ea_extension as ea
import run_ea_extension as queue
from project import save_json,digest
from verify_ea_bundle import verify as verify_original_bundle

AMENDMENT=ROOT/'protocol/ea_direct_launch_20261031.json'


def development_kind(argv,cwd,root=ROOT):
    """Exact command match; never match formal workers or other projects."""
    if Path(cwd).resolve()!=Path(root).resolve() or len(argv)<3:
        return None
    args=list(argv[1:])
    if args and args[0]=='-u':
        args=args[1:]
    if not args:
        return None
    script=Path(args[0])
    if not script.is_absolute():
        script=Path(cwd)/script
    script=script.resolve()
    if script==(Path(root)/'cloud/run_ea_extension.py').resolve() and args[1:]==['develop']:
        return 'development_queue'
    if script==(Path(root)/'src/ea_extension.py').resolve() and args[1:]==['--seed','2027','--development']:
        return 'development_worker'
    return None


def process_info(pid):
    try:
        p=Path('/proc')/str(pid)
        argv=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x]
        cwd=(p/'cwd').resolve(strict=True)
        # starttime is field22; suffix after final ')' begins at field3.
        start=(p/'stat').read_text().rsplit(')',1)[1].split()[19]
        kind=development_kind(argv,cwd)
        return dict(pid=int(pid),starttime=start,kind=kind,argv=argv) if kind else None
    except (OSError,ValueError,IndexError,UnicodeError):
        return None


def development_processes():
    return [info for p in Path('/proc').iterdir() if p.name.isdigit()
            if (info:=process_info(p.name)) is not None]


def signal_if_unchanged(info):
    current=process_info(info['pid'])
    if current==info:
        try:
            os.kill(info['pid'],signal.SIGTERM)
        except ProcessLookupError:
            pass


def stop_development():
    found=development_processes()
    parents=[p for p in found if p['kind']=='development_queue']
    workers=[p for p in found if p['kind']=='development_worker']
    if len(parents)>1 or len(workers)>1:
        raise RuntimeError('Multiple development tasks found; refusing ambiguous process stop')
    # The original queue's SIGTERM handler also terminates/reaps its worker.
    for p in parents or workers:
        print(f'Stopping excluded development task: PID={p["pid"]} {p["kind"]}',flush=True)
        signal_if_unchanged(p)
    deadline=time.monotonic()+45
    while development_processes():
        if time.monotonic()>deadline:
            raise RuntimeError('Development process has not exited; no formal job started')
        time.sleep(.5)
    print('DEVELOPMENT_STOPPED_OR_ALREADY_ABSENT; files preserved',flush=True)
    return found


def plan():
    _,_,spec=ea.load()
    amendment=json.loads(AMENDMENT.read_text(encoding='utf-8'))
    if amendment['formal_seeds']!=spec['formal_seeds'] or amendment['development_seed']!=spec['development_seed']:
        raise RuntimeError('Amendment seed roster mismatch')
    return dict(formal_seeds=spec['formal_seeds'],formal_runs=10,require_completed_development=False,
        development_seed_excluded=2027,training_sources_unchanged=True,cloud_preflight_required=True,
        test_scored=False,provenance=ea.provenance(),amendment_sha256=digest(AMENDMENT),
        launcher_sha256=digest(Path(__file__)))


def run_formal_tasks(spec):
    print('FORMAL_EA_QUEUE_STARTED: 10 fixed seeds; no full-development prerequisite',flush=True)
    for seed in spec['formal_seeds']:
        queue.launch(seed,False)
        print(f'Completed EA seed={seed}',flush=True)
    queue.collect(spec)


def run():
    record=plan()
    verify_original_bundle()
    if os.name!='posix':
        raise RuntimeError('Run only on the Linux cloud host')
    # Validate the complete preflight source identity before touching the old task.
    ready=json.loads((queue.PREFLIGHT/'ready.json').read_text(encoding='utf-8'))
    if ready['status']!='passed' or ready['provenance']!=ea.provenance():
        raise RuntimeError('Passed cloud EA preflight with unchanged sources required')
    stopped=stop_development()
    import fcntl
    def stop(signum,frame):
        raise KeyboardInterrupt(f'Received {signum}')
    signal.signal(signal.SIGTERM,stop)
    lock=ROOT/'runs/confirmation_queue.lock'
    with lock.open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        queue.require_idle_gpu()
        queue.check_ready()
        path=ea.OUT/'direct_launch_record.json'
        if path.exists():
            prior=json.loads(path.read_text(encoding='utf-8'))
            if prior['launch_plan']!=record:
                raise RuntimeError('Direct launch identity changed; refusing mixed queue')
        else:
            save_json(path,dict(launch_plan=record,stopped_development_processes=stopped,
                decision='User requested formal training without a complete excluded development run',
                development_artifacts_preserved=True))
        _,_,spec=ea.load()
        run_formal_tasks(spec)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['plan','run'])
    a=p.parse_args()
    if a.action=='plan':
        print(json.dumps(plan(),indent=2))
    else:
        run()
