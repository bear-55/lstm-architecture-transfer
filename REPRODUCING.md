# Reproduction guide

## 1. Validate the release

```bash
python reproduce.py verify
```

Checks are performed against `provenance/files.sha256.json`. The frozen training sources retain their original bytes; `.gitattributes` disables automatic line-ending conversion because SHA256 is part of the experiment contract. The release source map records both original and distributed hashes for each copied file. The feature dictionary and analysis protocols are published in English. The training-cache metadata records the digest of the translated feature notes together with the original dictionary digest. Training tensors, feature definitions, splits and weights retain their archived values. The external-protocol record identifies the original freeze and the English translation separately.

## 2. Recompute the statistical results

```bash
python reproduce.py results
```

The command copies the required numerical inputs and code into `work/analysis/`, then executes:

1. `scripts/matching_scale_followup.py`: paired uncertainty, scale/influence sensitivity and catchment matching.
2. `scripts/multimetric_transfer.py`: fixed-NSE and metric-specific choices, local/global decomposition and hydrological metrics.
3. `scripts/jh_predraft_baseline_audit.py`: main NSE intervals with the global architecture reselected within the bootstrap.
4. `scripts/multiseed_budget_curve_20261104.py`: all source budgets from one to five, disjoint evaluation pools, 20,000 regional bootstrap draws and paired budget changes.

The command also runs `scripts/selection_sensitivity.py` to reconstruct the seed-only and permutation results and Table S3. The report in `work/analysis/comparison.json` compares nine tables with the archived values. Supplying `--assets ../release-assets` also rebuilds the 103-catchment FLV sensitivity results for Table S7 from individual daily predictions. Figure-data preparation also rebuilds variance decompositions, trajectories, robustness matrices, matching summaries, fixed-library comparisons and seed contribution panels.

| Quantity | Reference file |
|---|---|
| Main held-out NSE endpoints | `outputs/jh_predraft_audit_20261102/main_NSE_endpoints.csv` |
| Multi-metric endpoints | `outputs/multimetric_transfer_20261102/transfer_endpoints.csv` |
| Per-catchment decomposition | `outputs/multimetric_transfer_20261102/transfer_per_basin.csv` |
| Source-budget endpoints and bounds | `outputs/multiseed_budget_curve_20261104/budget_curve_endpoints.csv` |
| Individual source-subset results | `outputs/multiseed_budget_curve_20261104/source_subset_results.csv` |
| Influence and matching results | `outputs/selection_transfer_P1_20261102/P1_endpoints.csv` |

`low95` and `high95` are the 2.5% and 97.5% percentile endpoints. `one_sided_upper95` is the 95% percentile. `source_subset_sd` describes variation across the enumerated, overlapping source subsets. Random seeds and resampling counts are fixed in the scripts and analysis specifications.

The score-level reproduction uses the archived observation-normalizer table. Its values were originally computed from daily observations; effective-day counts are checked against the individual-score records. The release validation additionally checks these normalizers against the daily-prediction attachment.

## 3. Reproduce the figures

```bash
python reproduce.py figures
python reproduce.py prepare-figures --assets ../release-assets
```

The first command uses the plot data in the repository. The second rebuilds numerical inputs and renders the same figures under `work/figure-data/figures/`. Figure 9 reads the archived daily predictions. The other figures can be prepared from the repository alone. Use `--figures 7` or `--figures S1 S2 S3` to select publication figures.

| Figure | Data and display |
|---|---|
| 1 | Regional model library, retained and independent weights, disjoint instance pools |
| 2 | Catchment roles, early B0 skill and catchment-wise retraining loss |
| 3 | Spatial persistence, independent-instance transfer and variance shares |
| 4 | Retained, transferred and global-rule gains |
| 5 | Source budgets of one to five instances, gain intervals and paired changes |
| 6 | Catchment-level gains across five hydrological metrics |
| 7 | Forest estimates and resampling densities, seven individual selection windows, regional matching and shares of the main estimate |
| 8 | Four fixed-library radar profiles, cumulative NSE gains and the three largest contributors |
| 9 | Daily hydrographs and flow-duration curves, including the mean and interquartile range of nine independent EA instances |
| S1 | Matching balance and early baseline skill distributions |
| S2 | Catchment-level gain decompositions across all seeds |
| S3 | Empirical power-law fits on linear plotting axes over the observed budgets |

The figure attachment contains the original 600-dpi PNG and TIFF files and the SVG and PDF files. Font substitution can change the appearance of a local render.

## 4. Inspect or retrain the models

Create a separate **Linux Python 3.12 GPU environment**, install `requirements-training.txt`, and extract `training-cache-v1.0.0.zip`. The archived environment used PyTorch **2.12.1+cu130**, NumPy **2.3.2**, CUDA 13.0 and an NVIDIA RTX 5090. The training code checks its required versions.

The cache and its hash dependencies are supplied. Source files and model/data hashes are checked before training. `configs/formal.json` is the inherited base specification; the current B0/M1 cohort replaces its seed list through `configs/retraining_confirmation.json`. EA uses `configs/ea_extension_20261031.json`. The current cohort comprises B0/M1/EA with the ten seeds in the README.

Use a fresh checkout and the training cache for a new training run. The archived model attachment belongs to the completed experiment and is useful for inspecting predictions/checkpoints. Existing completed runs have their original provenance and are sealed against accidental replacement.

```bash
python -m pip install -r requirements-training.txt
python cloud/run_retraining_confirmation.py plan
python cloud/run_retraining_confirmation.py preflight
python cloud/run_retraining_confirmation.py run
python cloud/run_ea_formal_direct.py plan
python cloud/run_ea_formal_direct.py run
```

The publication preparation reran the CPU analyses and all figure paths. GPU training is represented by the archived implementation, environment and completed 30-run library; no new training was performed during repository assembly.

## 5. External source archive

The score arrays for the external comparison are included under `outputs/weakness_resolution_20261102/external_NSE/` and `external_MSE/`. To retrieve the original authors' archive explicitly:

```bash
python scripts/download_weakness_external.py
```

The download script uses the public HydroShare resource and checks the recorded size and MD5. It is a separate network operation. The external run list, configuration inventory, temporal eligibility and original analysis specification accompany the derived scores.

## Release adaptation record

Model and training mathematics are unchanged. Publication copies of the analysis/plotting scripts replace machine-specific attribute paths with repository-relative paths; omit checksums of editorial Word files and the private plotting template; use an archived observation-normalizer table for score-level analysis; and support open-font fallbacks. The budget calculation keeps its complete subset enumeration and explicit checks of target-pool averaging and bootstrap global-rule reselection. Comparison against archived results is handled by `reproduce.py results`.
