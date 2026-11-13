# Numerical data guide

## Catchment identifiers and roles

Read `basin_id`, `gauge_id` and HUC codes as strings to retain leading zeros. The frozen partition has 370 `train`, 50 `val` and 107 `test` catchments. A HUC2 code identifies a two-digit US hydrologic region used for cluster resampling.

## Main score table

`deliverables/data_request_20261103/01_scores_long.csv` contains one record per scenario, period, architecture, seed and catchment. It includes the individual skill scores used for selection, transfer and figure preparation. Main scenario identifiers are `heldout107` and `trained370`. Model labels are `B0`, `M1` and `EA`. Period labels distinguish the original WY1990–1999 analysis, the aligned WY2000–2003 selection period and WY2004–2008 evaluation.

NSE and KGE are oriented so that larger values are better. `NSE_star` is the per-instance transform `NSE / (2 − NSE)`. Absolute FHV is measured in percentage points. Normalized high/low-flow MAE divides the per-instance error by the catchment's mean observed discharge in the same period. Error utilities use a negative sign before calculating gains. The figure preparation and metric definitions explicitly preserve this orientation.

## Metric arrays

`metric_arrays.npz` files contain `models`, `seeds`, `basins`, `regions`, `metrics`, `earlier` and `later`. The score arrays are indexed as **model × seed × catchment × metric**. Inspect the named arrays to establish ordering; do not infer row identity from file order.

## Endpoint tables

`estimate`, `low95` and `high95` give the point estimate and two-sided 95% percentile interval. Main endpoint labels `same`, `other`, `D`, `global_other` and `incremental_other` correspond to retained-weight gain, independently trained gain, their difference, transferred global-rule gain and local increment. Budget tables use the explicit names `G_same`, `G_new`, `G_global_new`, `D` and `increment_new`.

For the source-budget analysis, `source_budget` is the number of source instances; `target_budget` is five. `n_source_subsets` counts enumerated source subsets. At budget five, 252 directed allocations represent 126 complementary partitions. `one_sided_upper95` is a 95th percentile, distinct from `high95`. Source-subset dispersion is reported in `source_subset_sd`.

## Daily predictions and checkpoints

The model attachment retains the original run layout. `predictions_test.npz` contains aligned `basin_ids`, `dates`, `q_observed_mm_day` and `q_predicted_mm_day`. Discharge is in **mm/day**. All ten model instances are retained; daily series support hydrological diagnostics and recomputation of individual scores.

`best.pt` is the validation-selected checkpoint. `last.pt` stores the final training/resume state. `history.csv`, `run.json`, `result.json` and `completed_artifacts.json` document optimisation, configuration, completion and content hashes. Load checkpoints using PyTorch's restricted `weights_only=True` mode, as in the archived evaluation code.

## Figure inputs and published tables

Figure data and English captions are in `figures/Fig01` through `Fig09` and `FigS1` through `FigS4`. [The table index](tables/README.md) maps Tables 1–4 and the 14 supplementary tables to their numerical sources.

For Figure 8, `forest_endpoints.csv` identifies the interval or range for each row. The arrays in `forest_distributions.npz` use keys such as `M1__main` and `EA__k5`. The 21 densities use the same draws that define the corresponding intervals. Ordinary bootstrap and permutation rows use 2,000 draws, the WY1990–1999 rows use 5,000, and the five-source rows use 20,000 HUC2 draws. Each window is shown individually in panels a and b. Panel d uses their arithmetic mean. External-archive rows show point estimates and intervals.

In Figure 7, `daily_nine_instances.csv` and `fdc_nine_instances.csv` provide the plotted independent EA mean and quartiles. The plotted source is excluded separately in each catchment. Each independent hydrograph is sorted before estimating the flow-duration mean and quartiles. `independent_instance_scores.csv` identifies the nine evaluation seeds and their individual NSE scores.
