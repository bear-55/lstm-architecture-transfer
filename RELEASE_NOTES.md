# Reproducibility materials v1.0.0

This release accompanies *Retained-weight evaluation overstates the transferability of catchment-wise architecture choices in regional LSTM rainfall–runoff models*.

It provides the B0/M1/EA implementations and configurations, ten-seed experiment records, catchment splits, score-level analysis inputs, reference bootstrap outputs, data and plotting code for Figures 1–9 and S1–S3, and CSVs for the five main and thirteen supplementary tables.

## Attachments

- **model-artifacts-v1.0.0.zip** — 30 validation-selected checkpoints, resume checkpoints, training histories, run metadata, validation predictions and 150 period-specific evaluation-prediction files.
- **training-cache-v1.0.0.zip** — frozen model input cache, training/validation observations, scalers, metadata and held-out observation arrays.
- **publication-figures-v1.0.0.zip** — Figures 1–9 and S1–S3 in 600-dpi PNG/TIFF and vector SVG/PDF: 48 files.
- **SHA256SUMS.txt** — attachment checksums.

Download the three ZIP files to a sibling `release-assets` directory and run:

```bash
python -m pip install -r requirements.txt
python reproduce.py verify --assets ../release-assets
python reproduce.py results
python reproduce.py figures
```

The release checks the statistical analyses, the Table S3 interval calculations and the two figure workflows against the archived results. Figure 7 includes the bootstrap and permutation draws. Figure 9 includes the daily and flow-duration summaries for the nine independent EA instances. Validation details are in `provenance/release-validation.json`.

Code is MIT-licensed. CAMELS-US data and the external EA-LSTM simulations retain their source licences and attribution; see `DATA_SOURCES.md`.
