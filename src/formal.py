"""Portable formal-experiment entry point; reuses the tested pilot model/fit loop."""
import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from dataset import CamelsWindows
from project import ROOT, digest, save_json
from training import (WindowBatches, setup_device, environment, make_model, optimizer_for,
                      train_step, synchronize, predict_scaled)
from train import fit, identity_for


def cache_source_hashes(meta):
    """Apply an explicit seed-budget amendment without rewriting cache provenance."""
    expected = dict(meta["source_hashes"])
    revision_path = ROOT / "configs/formal_seed_revision.json"
    if revision_path.exists():
        revision = json.loads(revision_path.read_text(encoding="utf-8"))
        changes = revision["cache_source_hash_revisions"]
        allowed = {"configs/formal.json", "scripts/build_formal_dataset.py"}
        if set(changes) != allowed:
            raise RuntimeError("Seed amendment may only revise config and cache builder hashes")
        for name, hashes in changes.items():
            if expected.get(name) not in (hashes["before"], hashes["after"]):
                raise RuntimeError("Seed amendment does not match frozen source: " + name)
            expected[name] = hashes["after"]
    return expected


def load_formal():
    cfg = json.loads((ROOT / "configs/formal.json").read_text(encoding="utf-8"))
    meta = json.loads((ROOT / "data/formal/metadata.json").read_text(encoding="utf-8"))
    seeds = cfg["training"]["paired_seeds"]
    if (not seeds or len(set(seeds)) != len(seeds)
            or cfg["training"]["paired_seed_count"] != len(seeds)
            or cfg["training"]["confirmation_seeds"] != seeds):
        raise ValueError("Formal paired and confirmation seed lists/count must agree")
    for name, expected in cache_source_hashes(meta).items():
        if digest(ROOT / name) != expected:
            raise RuntimeError("Frozen formal input changed: " + name)
    if meta["test_q_in_cache"] or meta["test_q_read_by_this_builder"]:
        raise RuntimeError("Test Q must not enter formal training cache")
    for role, count in cfg["split_counts"].items():
        ids = (ROOT / f"splits/formal_pub_527/{role}.txt").read_text().split()
        assert ids == meta["splits"][role] and len(ids) == count
    assert len(set(sum(meta["splits"].values(), []))) == 527
    with np.load(ROOT / "data/formal/formal.npz", allow_pickle=False) as data:
        assert {x for x in data.files if x.startswith("q_")} == {"q_train", "q_val"}
    return cfg, meta


def provenance():
    files = ["configs/formal.json", "data/formal/metadata.json", "data/formal/formal.npz", "data/formal/scaler.json",
             "src/formal.py", "src/model.py", "src/dataset.py", "src/training.py", "src/train.py", "src/evaluate.py", "src/project.py"]
    return {x: digest(ROOT / x) for x in files}


def datasets(meta, device):
    train, val = [CamelsWindows(ROOT / "data/formal/formal.npz", meta, r) for r in ("train", "val")]
    return train, val, WindowBatches(train, device), WindowBatches(val, device)


def benchmark(cfg, meta, device, steps, output):
    if steps < 5:
        raise ValueError("At least five timing steps required")
    train, val, tb, vb = datasets(meta, device)
    size = cfg["training"]["batch_size"]
    indices = train.paired_epoch_indices(11, 0, (steps+3)*size)
    results = []
    for name in ("B0", "M1", "A1"):
        backend = "native_fused" if name == "B0" else "reference_manual_loop"
        model = make_model(cfg, meta, name, 11, device, backend)
        optimizer = optimizer_for(model, cfg)
        for i in range(3):
            train_step(model, optimizer, tb(indices[i*size:(i+1)*size]), cfg, meta["q_ref_mm_day"])
        synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        t = time.perf_counter()
        for i in range(3, steps+3):
            train_step(model, optimizer, tb(indices[i*size:(i+1)*size]), cfg, meta["q_ref_mm_day"])
        synchronize(device)
        train_step_s = (time.perf_counter()-t)/steps
        model.eval()
        t = time.perf_counter()
        with torch.no_grad():
            for i in range(steps):
                values = predict_scaled(model, vb(np.arange(i*size, (i+1)*size) % len(val)))
                assert torch.isfinite(values).all()
                values.cpu()
        synchronize(device)
        val_s = (time.perf_counter()-t)/steps*math.ceil(len(val)/size)
        row = {"model": name, "backend": backend, "training_step_seconds": train_step_s,
               "estimated_epoch_minutes": (train_step_s*math.ceil(len(train)/size)+val_s)/60,
               "peak_allocated_gpu_MiB": torch.cuda.max_memory_allocated(device)/2**20 if device.type == "cuda" else None}
        results.append(row)
        print(json.dumps(row), flush=True)
    save_json(output, {"environment": environment(device), "provenance": provenance(), "results": results,
              "train_windows": len(train), "val_windows": len(val), "steps": steps,
              "scope": "engineering throughput, no validation scores/checkpoints or Test Q"})


def run(args, cfg, meta, device):
    if args.seed not in cfg["training"]["confirmation_seeds"]:
        raise ValueError(f"Seed outside configured list: {cfg['training']['confirmation_seeds']}")
    directory = ROOT / f"runs/formal_pub_527/{args.model}/seed_{args.seed}"
    hashes = provenance()
    backend = "native_fused" if args.model == "B0" else "reference_manual_loop"
    identity = identity_for(args.model, args.seed, hashes, False, backend)
    resume = False
    if directory.exists():
        if not args.resume:
            raise RuntimeError("Run already exists; use --resume")
        spec = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        if spec["identity"] != identity:
            raise RuntimeError("Code/config/cache mismatch on resume")
        if (directory / "result.json").exists():
            print("Already completed: " + str(directory))
            return
        resume = (directory / "last.pt").exists()
    else:
        directory.mkdir(parents=True)
        save_json(directory / "run.json", {"identity": identity, "model": args.model, "seed": args.seed,
                  "provenance": hashes, "protocol": cfg, "backend": backend, "environment": environment(device),
                  "created_at": datetime.now(timezone.utc).isoformat()})
    train, val, tb, vb = datasets(meta, device)
    model = make_model(cfg, meta, args.model, args.seed, device, backend)
    fit(cfg, meta, model, optimizer_for(model, cfg), train, val, tb, vb, directory, identity, args.seed,
        device, resume=resume)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["check", "benchmark", "train"])
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--model", choices=["B0", "M1", "A1"], default="B0")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/formal_benchmark.json")
    args = parser.parse_args()
    cfg, meta = load_formal()
    device = setup_device(args.device)
    if str(torch.__version__).split("+")[0] != "2.12.1":
        raise RuntimeError("Use fixed PyTorch 2.12.1; do not mix training versions")
    if args.action == "check":
        tr, va, tb, _ = datasets(meta, device)
        batch = tb(tr.paired_epoch_indices(11, 0, 2))
        for name in ("B0", "M1", "A1"):
            model = make_model(cfg, meta, name, 11, device, "native_fused" if name == "B0" else "reference_manual_loop")
            loss = train_step(model, optimizer_for(model, cfg), batch, cfg, meta["q_ref_mm_day"])
            assert np.isfinite(loss)
        print(json.dumps({"status": "passed", "counts": cfg["split_counts"], "train_windows": len(tr),
              "val_windows": len(va), "environment": environment(device), "test_q_read": False}))
    elif args.action == "benchmark":
        benchmark(cfg, meta, device, args.steps, args.output)
    else:
        run(args, cfg, meta, device)


if __name__ == "__main__":
    main()
