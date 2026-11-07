"""Actual formal shapes: graph parity, tail batches, validation and RNG/optimizer replay."""
import argparse
import copy
import gc
import time
from pathlib import Path
import torch
import numpy as np
import formal
from graph_backend import attach_graphs
from project import save_json
from training import setup_device, make_model, optimizer_for, train_step, cpu_weights, rng_state, restore_rng, environment


def exact(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            exact(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            exact(x, y)
    else:
        assert a == b


def check_model(cfg, meta, name, device, tr, va, tb, vb):
    def new(seed):
        model = make_model(cfg, meta, name, seed, device)
        before = rng_state()
        attach_graphs(model, tb, vb, len(tr), len(va), 64)
        exact(before, rng_state())
        return model
    model = new(11)
    eager = make_model(cfg, meta, name, 11, device)
    optimizer, eo = optimizer_for(model, cfg), optimizer_for(eager, cfg)
    for n in (64, len(tr) % 64):
        batch = tb(np.arange(n))
        state = rng_state()
        loss_a = train_step(model, optimizer, batch, cfg, meta["q_ref_mm_day"])
        restore_rng(state)
        loss_b = train_step(eager, eo, batch, cfg, meta["q_ref_mm_day"])
        assert np.isclose(loss_a, loss_b, rtol=1e-5, atol=1e-6)
        for key in model.state_dict():
            torch.testing.assert_close(model.state_dict()[key], eager.state_dict()[key], rtol=1e-5, atol=1e-6)
    saved = {"model": cpu_weights(model), "optimizer": copy.deepcopy(optimizer.state_dict()), "rng": rng_state()}
    batch = tb(np.arange(64))
    loss = train_step(model, optimizer, batch, cfg, meta["q_ref_mm_day"])
    expected = {"model": cpu_weights(model), "optimizer": copy.deepcopy(optimizer.state_dict()), "rng": rng_state()}
    resumed = new(29)
    resumed.load_state_dict(saved["model"])
    ro = optimizer_for(resumed, cfg)
    ro.load_state_dict(saved["optimizer"])
    restore_rng(saved["rng"])
    resumed_loss = train_step(resumed, ro, batch, cfg, meta["q_ref_mm_day"])
    exact(loss, resumed_loss)
    exact(expected, {"model": cpu_weights(resumed), "optimizer": ro.state_dict(), "rng": rng_state()})
    resumed.eval()
    for n in (64, len(va) % 64):
        values = vb(np.arange(n))
        args = tuple(values[k] for k in ("x_dynamic", "x_static", "x_land"))
        with torch.no_grad():
            torch.testing.assert_close(resumed(*args), resumed._graph_backend.raw(*args), rtol=1e-5, atol=1e-6)
    times = {}
    for role, batches, count in (("train", tb, len(tr)), ("val", vb, len(va))):
        torch.cuda.synchronize()
        started = time.perf_counter()
        for i in range(20):
            values = batches(np.arange(i*64, (i+1)*64))
            if role == "train":
                train_step(resumed, ro, values, cfg, meta["q_ref_mm_day"])
            else:
                resumed.eval()
                with torch.no_grad():
                    resumed(values["x_dynamic"], values["x_static"], values["x_land"]).cpu()
        torch.cuda.synchronize()
        times[role + "_step_seconds"] = (time.perf_counter()-started)/20
    print(f"{name}: full/tail batches, validation, dropout and exact optimizer/RNG replay passed; {times}", flush=True)
    return {"model": name, "checks": "passed", **times}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    cfg, meta = formal.load_formal()
    device = setup_device("cuda")
    if str(torch.__version__).split("+")[0] != "2.12.1":
        raise RuntimeError("Expected PyTorch 2.12.1")
    tr, va, tb, vb = formal.datasets(meta, device)
    results = []
    for name in ("M1", "A1"):
        results.append(check_model(cfg, meta, name, device, tr, va, tb, vb))
        gc.collect()
        torch.cuda.empty_cache()
    from formal_graph import provenance
    save_json(args.output, {"status": "passed", "environment": environment(device), "provenance": provenance(), "results": results})


if __name__ == "__main__":
    main()
