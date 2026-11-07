"""Frozen S1/S2 training and bounded engineering benchmarks (never final-test Q)."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from evaluate import predict_validation, save_validation, write_csv
from model import parameter_count
from project import ROOT, save_json
from training import (load_protocol, setup_device, environment, provenance, make_model, datasets,
                      optimizer_for, train_step, synchronize, sampling_hash, rng_state, restore_rng,
                      cpu_weights, atomic_checkpoint)


def identity_for(name, seed, hashes, smoke, backend):
    spec = {"model": name, "seed": seed, "provenance": hashes, "smoke": smoke, "backend": backend}
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def fit(cfg, meta, model, optimizer, train, val, train_batches, val_batches, directory, identity, seed,
        device, *, resume=False, smoke=False, pause_after_epoch=None):
    """Epoch-atomic resume. Any interrupted partial epoch is replayed from its start."""
    max_epochs = 2 if smoke else cfg["training"]["max_epochs"]
    count = min(3 * cfg["training"]["batch_size"], len(train)) if smoke else len(train)
    history, best_score, best_epoch, bad_epochs, best_weights, start_epoch = [], -math.inf, None, 0, None, 0
    if resume:
        saved = torch.load(directory / "last.pt", map_location="cpu", weights_only=True)
        if saved["identity"] != identity:
            raise RuntimeError("Checkpoint identity mismatch")
        if saved["torch_version"] != str(torch.__version__) or saved["device_type"] != device.type:
            raise RuntimeError("Resume requires the same PyTorch version and CPU/CUDA device type")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        restore_rng(saved["rng"])
        history, best_score, best_epoch, bad_epochs, best_weights = (saved[key] for key in
            ("history", "best_score", "best_epoch", "bad_epochs", "best_model"))
        start_epoch = saved["completed_epochs"]
        # last.pt is authoritative if power failed between checkpoint and CSV writes.
        write_csv(directory / "history.csv", history)
    size = cfg["training"]["batch_size"]
    for epoch in range(start_epoch, max_epochs):
        if bad_epochs >= cfg["training"]["patience"]:
            break
        indices = train.paired_epoch_indices(seed, epoch, count)
        synchronize(device)
        started, weighted_loss = time.perf_counter(), 0.0
        n_steps = math.ceil(count / size)
        for step, offset in enumerate(range(0, count, size), 1):
            current = indices[offset:offset + size]
            loss = train_step(model, optimizer, train_batches(current), cfg, meta["q_ref_mm_day"])
            weighted_loss += loss * len(current)
            if step == 1 or step % 100 == 0 or step == n_steps:
                print(f"epoch {epoch+1}/{max_epochs} step {step}/{n_steps} loss={loss:.6f} elapsed={time.perf_counter()-started:.1f}s", flush=True)
        synchronize(device)
        train_seconds = time.perf_counter() - started
        val_started = time.perf_counter()
        rows, summary, records = predict_validation(model, val, val_batches, cfg, meta["q_ref_mm_day"])
        score = summary["median_nse"]
        if score is None:
            raise RuntimeError("No valid validation NSE; early stopping cannot select a model")
        improved = score > best_score  # Strict improvement; earliest tie is retained.
        if improved:
            best_score, best_epoch, bad_epochs = score, epoch + 1, 0
            best_weights = cpu_weights(model)
        else:
            bad_epochs += 1
        history.append({"epoch": epoch + 1, "train_loss": weighted_loss / count, "validation_median_nse": score,
                        "validation_mean_nse": summary["mean_nse"], "invalid_nse_basins": summary["invalid_nse_basins"],
                        "best_epoch": best_epoch, "bad_epochs": bad_epochs, "sampled_windows": count,
                        "optimizer_steps": n_steps, "sampling_sha256": sampling_hash(indices),
                        "training_seconds": train_seconds, "validation_seconds": time.perf_counter() - val_started})
        state = {"identity": identity, "torch_version": str(torch.__version__), "device_type": device.type,
                 "model": cpu_weights(model), "optimizer": optimizer.state_dict(), "rng": rng_state(),
                 "completed_epochs": epoch + 1, "history": history, "best_score": best_score,
                 "best_epoch": best_epoch, "bad_epochs": bad_epochs, "best_model": best_weights}
        atomic_checkpoint(directory / "last.pt", state)
        write_csv(directory / "history.csv", history)
        print(f"validation median NSE={score:.6f}; best epoch={best_epoch}; patience={bad_epochs}/{cfg['training']['patience']}", flush=True)
        if pause_after_epoch is not None and epoch + 1 == pause_after_epoch:
            return None  # Used by integration tests to simulate an epoch-boundary stop.
    if best_weights is None:
        raise RuntimeError("Training produced no selectable epoch")
    model.load_state_dict(best_weights)
    rows, summary, records = predict_validation(model, val, val_batches, cfg, meta["q_ref_mm_day"])
    if not np.isclose(summary["median_nse"], best_score, rtol=1e-6, atol=1e-6):
        raise RuntimeError("Reloaded best checkpoint does not reproduce its selection metric")
    atomic_checkpoint(directory / "best.pt", {"identity": identity, "model": best_weights, "epoch": best_epoch})
    save_validation(directory, rows, summary, records)
    result = {"status": "engineering_smoke_only" if smoke else "completed", "completed_epochs": len(history),
              "best_epoch": best_epoch, "stopped_by_patience": bad_epochs >= cfg["training"]["patience"],
              "validation": summary, "test_q_read": False, "identity": identity,
              "finished_at": str(datetime.now(timezone.utc))}
    save_json(directory / "result.json", result)
    return result


def run_training(args, cfg, meta, device):
    if args.seed not in cfg["training"]["confirmation_seeds"]:
        raise ValueError("Seed is outside the prespecified seed set")
    smoke = args.action == "smoke"
    directory = ROOT / "runs" / (".smoke_" + uuid.uuid4().hex if smoke else f"random_pub/{args.model}/seed_{args.seed}")
    hashes = provenance()
    backend = "native_fused" if args.model == "B0" and args.backend == "auto" else "reference_manual_loop"
    identity = identity_for(args.model, args.seed, hashes, smoke, backend)
    resume = getattr(args, "resume", False)
    if resume:
        previous = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        if previous["identity"] != identity or previous["provenance"] != hashes:
            raise RuntimeError("Resume refused: model, seed, code or frozen inputs changed")
        if (directory / "result.json").exists():
            print(f"Already completed: {directory}")
            return
        if not (directory / "last.pt").exists():
            # No complete epoch: restarting from the frozen initialization is exact.
            resume = False
    else:
        if directory.exists():
            raise FileExistsError(f"Run exists; use --resume: {directory}")
        directory.mkdir(parents=True)
        save_json(directory / "run.json", {"identity": identity, "model": args.model, "seed": args.seed,
                  "stage": "engineering" if smoke else "S1" if args.model == "B0" else "S2",
                  "provenance": hashes, "protocol": cfg, "environment": environment(device),
                  "backend": backend, "smoke": smoke, "created_at": str(datetime.now(timezone.utc))})
    try:
        train, val, train_batches, val_batches = datasets(meta, device)
        model = make_model(cfg, meta, args.model, args.seed, device, backend)
        print(f"{args.model} seed={args.seed} device={device} backend={backend} parameters={parameter_count(model)} output={directory}", flush=True)
        result = fit(cfg, meta, model, optimizer_for(model, cfg), train, val, train_batches, val_batches,
                     directory, identity, args.seed, device, resume=resume, smoke=smoke)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    finally:
        if smoke:
            resolved = directory.resolve()
            expected = (ROOT / "runs").resolve()
            if resolved.parent != expected or not resolved.name.startswith(".smoke_"):
                raise RuntimeError("Refusing cleanup outside the smoke directory")
            shutil.rmtree(resolved)


def benchmark(args, cfg, meta, device):
    train, val, train_batches, val_batches = datasets(meta, device)
    size = cfg["training"]["batch_size"]
    n_steps = args.steps
    if n_steps < 5:
        raise ValueError("Use at least 5 timed steps")
    indices = train.paired_epoch_indices(11, 0, (n_steps + 3) * size)
    results = []
    for name, fused in [("B0", False), ("M1", False), ("A1", False), ("B0", True)]:
        model = make_model(cfg, meta, name, 11, device)
        optimizer = optimizer_for(model, cfg)
        # Full-length, full-batch warmup including the optimizer, then timed steps.
        for i in range(3):
            train_step(model, optimizer, train_batches(indices[i*size:(i+1)*size]), cfg, meta["q_ref_mm_day"], fused_benchmark=fused)
        synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        for i in range(3, n_steps + 3):
            train_step(model, optimizer, train_batches(indices[i*size:(i+1)*size]), cfg, meta["q_ref_mm_day"], fused_benchmark=fused)
        synchronize(device)
        step_seconds = (time.perf_counter() - started) / n_steps
        model.eval()
        started = time.perf_counter()
        with torch.no_grad():
            for i in range(n_steps):
                batch = val_batches(np.arange(i*size, (i+1)*size) % len(val))
                output = model.fused_baseline(batch["x_dynamic"], batch["x_static"]) if fused else model(batch["x_dynamic"], batch["x_static"], batch["x_land"])
                if not torch.isfinite(output).all():
                    raise FloatingPointError("Nonfinite benchmark prediction")
                output.cpu()
        synchronize(device)
        validation_seconds = (time.perf_counter() - started) / n_steps * math.ceil(len(val) / size)
        epoch_seconds = step_seconds * math.ceil(len(train) / size) + validation_seconds
        row = {"model": name, "backend": "native_fused_timing_only" if fused else "reference_manual_loop",
               "training_step_seconds": step_seconds, "estimated_validation_seconds": validation_seconds,
               "estimated_epoch_minutes": epoch_seconds / 60, "estimated_30_epoch_hours": epoch_seconds * 30 / 3600,
               "peak_allocated_gpu_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None}
        results.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    report = {"measured_at": str(datetime.now(timezone.utc)), "environment": environment(device), "provenance": provenance(),
              "batch_size": size, "sequence_length": cfg["model"]["seq_length"], "warmup_steps": 3, "timed_steps": n_steps,
              "training_windows_per_epoch": len(train), "validation_windows": len(val), "results": results,
              "scope": "Engineering throughput only; no checkpoint, validation score or final-test Q. Estimates exclude startup/checkpoint IO and early stopping.",
              "formal_backend": {"B0": "native_fused", "M1": "reference_manual_loop", "A1": "reference_manual_loop"}}
    save_json(ROOT / "runs/s0/benchmark.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("train", "smoke", "benchmark"):
        child = sub.add_parser(action)
        child.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
        if action == "benchmark":
            child.add_argument("--steps", type=int, default=20)
        else:
            child.add_argument("--model", choices=["B0", "M1", "A1"], default="B0")
            child.add_argument("--seed", type=int, default=11)
            child.add_argument("--backend", choices=["auto", "manual"], default="auto",
                               help="auto: native B0, reference loop M1/A1; manual: reference loop for all")
            if action == "train":
                child.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg, meta = load_protocol()
    device = setup_device(args.device)
    if args.action == "benchmark":
        benchmark(args, cfg, meta, device)
    else:
        run_training(args, cfg, meta, device)


if __name__ == "__main__":
    main()
