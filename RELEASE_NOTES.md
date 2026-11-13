# Reproducibility materials v1.2.0

This release accompanies *Retained-weight evaluation overstates transferable gains from catchment-wise architecture selection in regional LSTM rainfall–runoff models*.

It provides the B0/M1/EA implementations and configurations, ten-seed experiment records, catchment splits, score-level analysis inputs, reference bootstrap outputs, data and plotting code for Figures 1–9 and S1–S4, and CSVs for the four main and fourteen supplementary tables.

Section 4.2 and Figure 9 compare observed retraining losses with Eq. (7) across six settings. The source archive includes the 256-unit external score table, per-catchment calculations and regional summaries for both training objectives.

Figure 2 now shows the climate strata and early baseline skill distributions for all three catchment groups. The hydrographs, robustness checks and Eq. (7) analysis are Figures 7, 8 and 9. Fixed-library deployment is Figure S4 and Table S12. Captions follow the submission manuscript and supplement.

## Downloads

- [model-artifacts-v1.0.0.zip](https://github.com/bear-55/lstm-architecture-transfer/releases/download/v1.0.0/model-artifacts-v1.0.0.zip) — 30 validation-selected checkpoints, resume checkpoints, training histories, run metadata, validation predictions and 150 period-specific evaluation-prediction files.
- [training-cache-v1.0.0.zip](https://github.com/bear-55/lstm-architecture-transfer/releases/download/v1.0.0/training-cache-v1.0.0.zip) — frozen model input cache, training/validation observations, scalers, metadata and held-out observation arrays.
- [publication-figures-v1.2.0.zip](https://github.com/bear-55/lstm-architecture-transfer/releases/download/v1.2.0/publication-figures-v1.2.0.zip) — Figures 1–9 and S1–S4 in 600-dpi PNG/TIFF and vector SVG/PDF: 52 files.
- [SHA256SUMS.txt](https://github.com/bear-55/lstm-architecture-transfer/releases/download/v1.2.0/SHA256SUMS.txt) — attachment checksums.

Download the three ZIP files to a sibling `release-assets` directory and run:

```bash
python -m pip install -r requirements.txt
python reproduce.py verify --assets ../release-assets
python reproduce.py results
python reproduce.py figures
```

The release checks the statistical analyses, the Table S3 interval calculations and the two figure workflows against the archived results. Figure 8 includes the bootstrap and permutation draws. Figure 7 includes the daily and flow-duration summaries for the nine independent EA instances. Validation details are in `provenance/release-validation.json`.

Code is MIT-licensed. CAMELS-US data and the external EA-LSTM simulations retain their source licences and attribution; see `DATA_SOURCES.md`.
