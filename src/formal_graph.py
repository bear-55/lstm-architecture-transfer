"""Formal CUDA Graph entry; explicitly records migration from the original backend."""
import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
import torch
import formal
from graph_backend import attach_graphs
from project import ROOT, digest, save_json
from train import fit, identity_for
from training import make_model, setup_device, optimizer_for, environment, atomic_checkpoint


def provenance():
    result = formal.provenance()
    for name in ("src/graph_backend.py", "src/formal_graph.py", "src/graph_checks.py", "cloud/run_graph_queue.py"):
        result[name] = digest(ROOT / name)
    return result


def migrate_legacy(name, seed, target, identity):
    source = ROOT / f"runs/formal_pub_527/{name}/seed_{seed}"
    if not (source / "run.json").exists():
        return
    previous = json.loads((source / "run.json").read_text(encoding="utf-8"))
    backend = "native_fused" if name == "B0" else "reference_manual_loop"
    expected = identity_for(name, seed, formal.provenance(), False, backend)
    if previous["identity"] != expected or previous["provenance"] != formal.provenance():
        raise RuntimeError("Legacy run does not match frozen source/config/data")
    record = {"source": str(source), "source_identity": expected, "target_identity": identity,
              "old_backend": backend, "new_backend": "native_fused" if name == "B0" else "cuda_graph",
              "time": datetime.now(timezone.utc).isoformat(), "completed_epochs_reused": 0,
              "partial_epoch_policy": "replay from last complete epoch; originals retained"}
    for name_pt in ("last.pt", "best.pt"):
        path = source / name_pt
        if not path.exists():
            continue
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["identity"] != expected:
            raise RuntimeError("Legacy checkpoint identity mismatch")
        if name_pt == "last.pt":
            if saved["torch_version"] != str(torch.__version__) or saved["device_type"] != "cuda":
                raise RuntimeError("Cannot migrate across PyTorch/device type")
            record["completed_epochs_reused"] = saved["completed_epochs"]
        record[name_pt + "_source_sha256"] = digest(path)
        saved["identity"] = identity
        saved["migration_source_identity"] = expected
        atomic_checkpoint(target / name_pt, saved)
    for name_file in ("history.csv", "metrics_per_basin.csv", "predictions_val.npz", "validation.json"):
        if (source / name_file).exists():
            shutil.copy2(source / name_file, target / name_file)
    if (source / "result.json").exists():
        if not (target / "best.pt").exists() or not (target / "last.pt").exists():
            raise RuntimeError("Completed legacy run lacks checkpoint files")
        result = json.loads((source / "result.json").read_text(encoding="utf-8"))
        if result["identity"] != expected:
            raise RuntimeError("Legacy result identity mismatch")
        result.update(identity=identity, migration_source_identity=expected,
                      completed_with_backend=backend)
        save_json(target / "result.json", result)
    save_json(target / "migration.json", record)
    print(json.dumps(record), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["B0", "M1", "A1"])
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg, meta = formal.load_formal()
    device = setup_device("cuda")
    if str(torch.__version__).split("+")[0] != "2.12.1":
        raise RuntimeError("Expected PyTorch 2.12.1")
    if args.seed not in cfg["training"]["confirmation_seeds"]:
        raise ValueError("Unexpected seed")
    target = ROOT / f"runs/formal_pub_527_graph/{args.model}/seed_{args.seed}"
    hashes = provenance()
    backend = "native_fused" if args.model == "B0" else "cuda_graph"
    identity = identity_for(args.model, args.seed, hashes, False, backend)
    if target.exists():
        if not args.resume:
            raise RuntimeError("Existing run requires --resume")
        spec = json.loads((target / "run.json").read_text(encoding="utf-8"))
        if spec["identity"] != identity:
            raise RuntimeError("Graph run code/data/config changed")
    else:
        target.mkdir(parents=True)
        save_json(target / "run.json", {"identity": identity, "provenance": hashes, "model": args.model,
                  "seed": args.seed, "backend": backend, "protocol": cfg, "environment": environment(device)})
    # Complete any migration interrupted before its completion record was written.
    if not (target / "migration.json").exists() and not (target / "graph_run_started.json").exists():
        migrate_legacy(args.model, args.seed, target, identity)
    if (target / "result.json").exists():
        print("Already completed: " + str(target), flush=True)
        return
    save_json(target / "graph_run_started.json", {"backend": backend})
    tr, va, tb, vb = formal.datasets(meta, device)
    model = make_model(cfg, meta, args.model, args.seed, device,
                       "native_fused" if args.model == "B0" else "reference_manual_loop")
    attach_graphs(model, tb, vb, len(tr), len(va), cfg["training"]["batch_size"])
    print(f"{args.model} seed={args.seed} backend={backend}; output={target}", flush=True)
    fit(cfg, meta, model, optimizer_for(model, cfg), tr, va, tb, vb, target, identity, args.seed,
        device, resume=(target / "last.pt").exists())


if __name__ == "__main__":
    main()
