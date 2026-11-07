"""Single-GPU confirmation queue. No Test scoring or old checkpoint migration."""
import argparse
import csv
import gc
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import retraining_confirmation as cohort
from project import save_json, digest

PREFLIGHT = ROOT / 'runs/confirmation_preflight'
LOGS = ROOT / 'runs/confirmation_logs'


def gpu_environment():
    return dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')


def require_idle_gpu():
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    if result.stdout.strip():
        raise RuntimeError('A GPU compute process is already running: ' + result.stdout.strip())


def witness():
    import numpy as np
    import torch
    import formal
    from graph_checks import check_model, exact
    from training import setup_device, make_model, optimizer_for, train_step, rng_state, restore_rng, cpu_weights
    import copy
    cohort.check_versions()
    cfg, meta, _ = cohort.load()
    device = setup_device('cuda')
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Expected exactly one visible GPU')
    torch.manual_seed(0)
    x = torch.randn(32, 32, device=device)
    y = x @ x
    torch.cuda.synchronize()
    if not torch.isfinite(y).all().item():
        raise RuntimeError('CUDA kernel witness failed')
    print('CUDA_KERNEL_WITNESS_OK', flush=True)
    tr, va, tb, vb = formal.datasets(meta, device)
    size = cfg['training']['batch_size']
    # B0 native/eager update parity and exact native replay at formal shape.
    b0 = make_model(cfg, meta, 'B0', 1009, device, 'native_fused')
    eager = make_model(cfg, meta, 'B0', 1009, device)
    opt, eo = optimizer_for(b0, cfg), optimizer_for(eager, cfg)
    for n in sorted({size, len(tr) % size} - {0}):
        batch = tb(np.arange(n))
        state = rng_state()
        a = train_step(b0, opt, batch, cfg, meta['q_ref_mm_day'])
        restore_rng(state)
        b = train_step(eager, eo, batch, cfg, meta['q_ref_mm_day'])
        if not np.isclose(a, b, rtol=1e-4, atol=1e-5):
            raise RuntimeError('B0 fused/eager loss mismatch')
    saved = dict(model=cpu_weights(b0), optimizer=copy.deepcopy(opt.state_dict()), rng=rng_state())
    batch = tb(np.arange(size))
    expected_loss = train_step(b0, opt, batch, cfg, meta['q_ref_mm_day'])
    expected = dict(model=cpu_weights(b0), optimizer=copy.deepcopy(opt.state_dict()), rng=rng_state())
    replay = make_model(cfg, meta, 'B0', 1021, device, 'native_fused')
    replay.load_state_dict(saved['model'])
    ro = optimizer_for(replay, cfg)
    ro.load_state_dict(saved['optimizer'])
    restore_rng(saved['rng'])
    exact(expected_loss, train_step(replay, ro, batch, cfg, meta['q_ref_mm_day']))
    exact(expected, dict(model=cpu_weights(replay), optimizer=ro.state_dict(), rng=rng_state()))
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(5):
        train_step(replay, ro, batch, cfg, meta['q_ref_mm_day'])
    torch.cuda.synchronize()
    b0_seconds = (time.perf_counter() - started) / 5
    print('B0_FULL_TAIL_AND_REPLAY_OK', flush=True)
    del b0, eager, replay, opt, eo, ro, expected, saved, batch
    gc.collect()
    torch.cuda.empty_cache()
    m1 = check_model(cfg, meta, 'M1', device, tr, va, tb, vb)
    save_json(PREFLIGHT / 'ready.json', dict(status='passed', provenance=cohort.provenance(),
        environment=cohort.runtime(device), train_windows=len(tr), validation_windows=len(va),
        B0_train_step_seconds=b0_seconds, M1=m1, test_scored=False))
    print('PREFLIGHT_PASSED: B0/M1 only; no experiment checkpoints or Test scoring', flush=True)


def check_ready():
    from training import setup_device
    cohort.check_versions()
    ready = json.loads((PREFLIGHT / 'ready.json').read_text())
    if (ready['status'] != 'passed' or ready['provenance'] != cohort.provenance()
            or ready['environment'] != cohort.runtime(setup_device('cuda'))):
        raise RuntimeError('Preflight code/data/environment changed; rerun preflight')


def preflight():
    require_idle_gpu()
    PREFLIGHT.mkdir(parents=True, exist_ok=True)
    (PREFLIGHT / 'ready.json').unlink(missing_ok=True)
    log = PREFLIGHT / 'preflight.log'
    print(f'Preflight started. Log: {log}', flush=True)
    with log.open('w', encoding='utf-8') as f:
        subprocess.run([sys.executable, '-u', __file__, '_witness'], cwd=ROOT,
                       env=gpu_environment(), stdout=f, stderr=subprocess.STDOUT, check=True)
    print(log.read_text(encoding='utf-8'), flush=True)


def status(spec):
    states = []
    for m, s in cohort.tasks(spec):
        p = cohort.OUT / m / f'seed_{s}'
        if (p / 'result.json').exists():
            r = json.loads((p / 'result.json').read_text())
            label = f'completed ({r["completed_epochs"]} epochs)'
        elif (p / 'last.pt').exists():
            label = 'checkpoint present, unfinished'
        elif (p / 'run.json').exists():
            label = 'started, no completed epoch'
        else:
            label = 'pending'
        states.append(f'{m:2} seed={s}: {label}')
    print('\n'.join(states))


def collect(cfg, spec):
    import torch
    hashes = cohort.provenance()
    histories, files = {}, {}
    for m, s in cohort.tasks(spec):
        p = cohort.OUT / m / f'seed_{s}'
        identity = cohort.identity_for(m, s, hashes, False, cohort.backend(m))
        if not cohort.check_existing(p, m, s, cfg, hashes, identity):
            raise RuntimeError(f'Incomplete run: {m}/{s}')
        cohort.validate_completed(p, identity)
        with (p / 'history.csv').open(newline='') as f:
            histories[m, s] = list(csv.DictReader(f))
        for name in cohort.REQUIRED + ('completed_artifacts.json',):
            files[(p / name).relative_to(cohort.OUT).as_posix()] = digest(p / name)
    for s in spec['new_seeds']:
        first, second = (histories[m, s] for m in cohort.MODELS)
        for a, b in zip(first, second):
            for field in ('sampling_sha256', 'sampled_windows', 'optimizer_steps'):
                if a[field] != b[field]:
                    raise RuntimeError(f'Paired sampling mismatch: seed={s}, epoch={a["epoch"]}')
    save_json(cohort.OUT / 'cohort_completed.json', dict(status='completed', runs=20,
        new_seeds=spec['new_seeds'], paired_sampling_checked=True, test_scored=False,
        provenance=hashes, files=files))
    print('ALL_20_RUNS_COMPLETED_AND_VERIFIED. Test evaluation remains separate.', flush=True)


def run_queue(cfg, spec):
    require_idle_gpu()
    check_ready()
    LOGS.mkdir(parents=True, exist_ok=True)
    child = None
    try:
        for model, seed in cohort.tasks(spec):
            path = LOGS / f'{model}_seed_{seed}.log'
            with path.open('a', encoding='utf-8') as log:
                print(f'Started {model} seed={seed}; GPU=0; log={path}', flush=True)
                child = subprocess.Popen([sys.executable, '-u', 'src/retraining_confirmation.py',
                    '--model', model, '--seed', str(seed), '--resume'], cwd=ROOT,
                    env=gpu_environment(), stdout=log, stderr=subprocess.STDOUT)
                code = child.wait()
                child = None
            if code:
                raise RuntimeError(f'{model} seed={seed} failed (exit={code}); see {path}; queue stopped')
            print(f'Completed {model} seed={seed}', flush=True)
        collect(cfg, spec)
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'preflight', 'run', 'status', 'collect', '_witness'))
    args = parser.parse_args()
    cfg, _, spec = cohort.load()
    if args.action == 'plan':
        print(json.dumps(dict(models=list(cohort.MODELS), seeds=spec['new_seeds'],
            jobs=cohort.tasks(spec), run_count=20, gpu='0', serial=True,
            output=str(cohort.OUT), test_labels_used=False), indent=2))
        return
    if args.action == 'status':
        status(spec)
        return
    if args.action == '_witness':
        witness()
        return
    if os.name != 'posix':
        raise RuntimeError('Launch on the Linux cloud host; plan/status are portable')
    import fcntl
    def stop(signum, frame):
        raise KeyboardInterrupt(f'Received signal {signum}')
    signal.signal(signal.SIGTERM, stop)
    lock_path = ROOT / 'runs/confirmation_queue.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == 'preflight':
            preflight()
        elif args.action == 'run':
            run_queue(cfg, spec)
        else:
            collect(cfg, spec)


if __name__ == '__main__':
    main()
