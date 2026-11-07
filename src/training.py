"""Shared, development-only training machinery for the frozen pilot."""
from __future__ import annotations

import hashlib
import json
import os
import platform
from pathlib import Path

import numpy as np
import torch

from dataset import CamelsWindows
from model import LandMemoryLSTM, basin_scaled_mse, copy_base_weights
from project import ROOT, config, digest, split_ids


def load_protocol():
    cfg = config()
    meta = json.loads((ROOT / "data/processed/metadata.json").read_text(encoding="utf-8"))
    files = {"config_hash": "configs/pilot.yaml", "feature_manifest_hash": cfg["features"]["manifest"],
             "selection_manifest_hash": "splits/selection_manifest.json", "cache_sha256": "data/processed/pilot.npz",
             "scaler_sha256": "data/processed/scaler.json"}
    for key, relative in files.items():
        if meta[key] != digest(ROOT / relative):
            raise RuntimeError(f"Frozen input changed: {relative}")
    if meta["splits"] != split_ids() or meta["test_q_read"] or meta["test_q_in_cache"]:
        raise RuntimeError("Split or test-label boundary changed")
    with np.load(ROOT / "data/processed/pilot.npz", allow_pickle=False) as saved:
        if any(key.startswith("q_") and key not in ("q_train", "q_val") for key in saved.files):
            raise PermissionError("Unexpected labels in development cache")
    s0 = json.loads((ROOT / "runs/s0/checks.json").read_text(encoding="utf-8"))
    if s0["status"] != "passed" or s0["protocol_hash"] != meta["config_hash"] or s0["cache_sha256"] != meta["cache_sha256"]:
        raise RuntimeError("S0 must pass for these exact inputs")
    if s0["model_sha256"] != digest(ROOT / "src/model.py"):
        raise RuntimeError("Reference model changed after S0")
    if cfg["evaluation"]["read_final_test_q"] or cfg["evaluation"]["final_protocol_frozen"]:
        raise PermissionError("This entry point only supports the development protocol")
    return cfg, meta


def setup_device(name="auto"):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda" if name == "auto" and torch.cuda.is_available() else "cpu" if name == "auto" else name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; check the GPU environment")
    return device


def environment(device):
    return {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__,
            "platform": platform.platform(), "device_type": device.type,
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
            "precision": "float32; TF32 disabled", "deterministic_algorithms": True, "cpu_threads": 1}


def provenance():
    names = ["src/model.py", "src/dataset.py", "src/project.py", "src/training.py", "src/train.py", "src/evaluate.py",
             "configs/pilot.yaml", "data/processed/metadata.json", "data/processed/scaler.json", "data/processed/pilot.npz",
             "data_contract/feature_manifest.csv", "splits/selection_manifest.json", "pyproject.toml", "uv.lock"]
    return {name: digest(ROOT / name) for name in names}


def make_model(cfg, meta, name, seed, device, backend="reference_manual_loop"):
    if name not in cfg["model"]["models"]:
        raise ValueError("Only B0, M1 and A1 are in this pilot")
    torch.manual_seed(seed)
    sizes = dict(dynamic_size=len(meta["dynamic_names"]), static_size=len(meta["static_names"]),
                 land_size=len(meta["land_names"]), hidden_size=cfg["model"]["hidden_size"],
                 rank=cfg["model"]["rank"], dropout=cfg["model"]["output_dropout"])
    base = LandMemoryLSTM(**sizes, adapter_mode="none")
    model = LandMemoryLSTM(**sizes, adapter_mode=cfg["model"]["models"][name])
    copy_base_weights(base, model)
    model = model.to(device)
    if backend not in ("reference_manual_loop", "native_fused") or (backend == "native_fused" and name != "B0"):
        raise ValueError("Native fused execution is only valid for B0")
    model.execution_backend = backend
    # Module construction consumes different amounts of RNG. Reset dropout's stream.
    torch.manual_seed(seed + 100000)
    return model


class WindowBatches:
    """Vectorized gathering, identical to CamelsWindows.__getitem__, no new sampling."""
    def __init__(self, dataset, device):
        self.dataset, self.device = dataset, device
        self.basins = np.repeat(np.arange(len(dataset.ids)), np.diff(np.r_[0, dataset.cumulative]))
        self.ends = np.concatenate(dataset.ends)
        self.rows = np.asarray(dataset.basin_rows)[self.basins]
        self.lags = np.arange(1 - dataset.length, 1)

    def __call__(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1 or len(indices) == 0 or indices.min() < 0 or indices.max() >= len(self.dataset):
            raise IndexError("Invalid window indices")
        ds = self.dataset
        rows, ends, basins = self.rows[indices], self.ends[indices], self.basins[indices]
        dynamic = ds.dynamic[rows[:, None], ends[:, None] + self.lags]
        static = ds.static[rows]
        arrays = {"x_dynamic": dynamic, "x_static": static, "x_land": static[:, ds.land_indices],
                  "q_obs_mm": ds.q[basins, ends, None]}
        if ds.role == "train":
            arrays["sigma_train_mm"] = np.asarray([ds.sigma[ds.ids[j]] for j in basins], dtype=np.float32)[:, None]
        if not all(np.isfinite(value).all() for value in arrays.values()):
            raise FloatingPointError("Nonfinite input/label/scale in a prevalidated window")
        if (arrays["q_obs_mm"] < 0).any() or (ds.role == "train" and (arrays["sigma_train_mm"] < 0).any()):
            raise ValueError("Negative Q or source sigma")
        return {key: torch.from_numpy(np.ascontiguousarray(value)).to(self.device) for key, value in arrays.items()}


def datasets(meta, device):
    cache = ROOT / "data/processed/pilot.npz"
    train, val = (CamelsWindows(cache, meta, role) for role in ("train", "val"))
    return train, val, WindowBatches(train, device), WindowBatches(val, device)


def optimizer_for(model, cfg):
    return torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"])


def predict_scaled(model, batch):
    if getattr(model, "execution_backend", "reference_manual_loop") == "native_fused":
        if model.adapter_mode != "none":
            raise ValueError("Never bypass the adapter in a trained candidate")
        return model.fused_baseline(batch["x_dynamic"], batch["x_static"])
    return model(batch["x_dynamic"], batch["x_static"], batch["x_land"])


def train_step(model, optimizer, batch, cfg, q_ref, *, fused_benchmark=False):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    if fused_benchmark and model.adapter_mode != "none":
        raise ValueError("Fused timing only applies to unmodified B0")
    prediction = model.fused_baseline(batch["x_dynamic"], batch["x_static"]) if fused_benchmark else predict_scaled(model, batch)
    loss = basin_scaled_mse(prediction * q_ref, batch["q_obs_mm"], batch["sigma_train_mm"], epsilon=cfg["training"]["epsilon_mm_day"])
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite training loss")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip_norm"], error_if_nonfinite=True)
    optimizer.step()
    return loss.item()


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def sampling_hash(indices):
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def rng_state():
    return {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}


def restore_rng(state):
    torch.set_rng_state(state["cpu"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def cpu_weights(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def atomic_checkpoint(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)
