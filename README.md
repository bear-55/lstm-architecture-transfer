# LSTM architecture transfer

Reproducibility materials for **Retained-weight evaluation overstates the transferability of catchment-wise architecture choices in regional LSTM rainfall–runoff models**.

Xiong Wu, Xiaoran Fu, Jinlun Jiang, Haijia Zhang, Hongquan Sun and Zhonggen Wang.

This repository evaluates whether a catchment-wise architecture choice transfers to independently trained instances. Historical choices are fixed, then evaluated with retained weights and with different training seeds. The experiment contains **30 fitted models**: ten instances each of a concatenation LSTM (B0), a conditional forget-gate variant (M1), and an Entity-Aware LSTM (EA).

## Quick start

Use Python **3.12**. The CPU analysis environment is separate from the archived GPU training environment.

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py verify
python reproduce.py results
python reproduce.py figures
```

`verify` checks file hashes, the frozen training implementation, catchment membership, ten-seed identities, figure inputs and independently enumerated main point estimates. `results` recomputes the main intervals, multi-metric results, sensitivity and matching results, the source-budget curves and Table S3, comparing them with the archived reference tables. `figures` draws the nine main figures and three supplementary figures from the included figure data. These three commands use the included score tables and run on CPU.

Recomputed results are written to `work/analysis/`; figures under `work/figures/figures/` with the same main and supplementary IDs as the paper. Each figure is exported as **600-dpi PNG and TIFF**, plus **vector SVG and PDF**. Times New Roman is used when installed, with open-font fallbacks. To redraw Figure 7 and Supplementary Figure S1:

```bash
python reproduce.py figures --figures 7 S1
```

## Experiment

| Component | Setting |
|---|---|
| Catchments | 370 training, 50 validation, 107 held-out gauged CAMELS-US catchments |
| Architectures | B0, M1 and EA; 64 hidden units; ten instances per architecture |
| Seeds | 1009, 1021, 1033, 1051, 1063, 1087, 1097, 1109, 1123, 1151 |
| Training labels | WY1990–1999 |
| Main selection period | WY2000–2003 |
| Evaluation period | WY2004–2008 |
| Prespecified selection analysis | WY1990–1999, retained separately in the archive |
| Multi-instance selection | One to five source instances; five disjoint evaluation instances |

The retained-weight gain is `G_same`; the gain transferred to a different instance is `G_new`. Their difference is `D = G_same − G_new`. The local increment is `G_new − G_global,new`, where the global architecture is selected using the same historical information. At a source budget of five, all **252 directed allocations** are included. Evaluation averages the five individual scores.

The dated analysis records distinguish the original selection analysis from the exploratory extensions. Ordinary main intervals resample seeds and HUC2 regions. Source-budget intervals resample HUC2 regions conditional on the ten-instance library; source subsets remain together, and the global rule is reselected within each resample. Details and command mapping are in [REPRODUCING.md](REPRODUCING.md).

## Repository contents

| Path | Contents |
|---|---|
| `src/`, `cloud/`, `configs/` | Frozen model, training and evaluation implementation |
| `analysis/`, `scripts/` | Selection/transfer calculations and statistical analyses |
| `splits/`, `data_contract/` | Frozen catchment assignments and feature definitions |
| `deliverables/data_request_20261103/` | Seed-specific score tables, metadata and run summaries |
| `outputs/` | Numerical reference results and bootstrap draws |
| [tables/](tables/) | Tables 1–5 and all 13 supplementary tables, with their source data |
| `figures/Fig01`–`Fig09` and `figures/FigS1`–`FigS3` | Plotting code, numerical plot inputs and concise English captions |
| `protocol/`, `provenance/` | Dated analysis records, source hashes and release validation |
| `release-assets.json` | SHA256 and per-file inventories for the three release attachments |

Historical directory names in the numerical archive are stable identifiers. The current entry point is `reproduce.py`.

## Release attachments

Download these attachments from the matching GitHub release into a sibling `release-assets` directory:

- `model-artifacts-v1.0.0.zip`: all 30 selected checkpoints, final/resume checkpoints, histories, run metadata and seed-specific daily predictions.
- `training-cache-v1.0.0.zip`: the frozen training/validation cache, scalers, metadata and held-out observation arrays.
- `publication-figures-v1.0.0.zip`: the 48 final figure files used for the manuscript, four formats for each of Figures 1–9 and S1–S3.

Verify the ZIPs and optionally extract them at repository root:

```bash
python reproduce.py verify --assets ../release-assets
python reproduce.py extract --assets ../release-assets
```

To rebuild the numerical inputs of all figures, including the daily hydrographs:

```bash
python reproduce.py prepare-figures --assets ../release-assets
```

The original external EA-LSTM simulations are available from their authors through [HydroShare](https://doi.org/10.4211/hs.83ea5312635e44dc824eeb99eda12f06). Their derived scores are included here. Data origins, transformations and licence attribution are listed in [DATA_SOURCES.md](DATA_SOURCES.md).

## Citation and licence

Software citation metadata are in [CITATION.cff](CITATION.cff). Code is released under the [MIT licence](LICENSE). CAMELS-US and the external simulations retain their source licences and citations; see [DATA_SOURCES.md](DATA_SOURCES.md). The repository URL and archival DOI can be added to the citation metadata after publication.
