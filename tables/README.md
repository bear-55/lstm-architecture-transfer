# Published tables

The CSV files contain the English table bodies at the precision used in the paper. Section labels from merged table rows are carried into separate columns. Full-precision results are linked below.

| Table | Contents | Data |
|---|---|---|
| [Table 1](Table_1.csv) | Catchments, model configurations and periods | [Source](../splits/formal_pub_527/) |
| [Table 2](Table_2.csv) | Single-instance gains and retraining loss | [Source](../outputs/jh_predraft_audit_20261102/) |
| [Table 3](Table_3.csv) | Source budget and local increments | [Source](../outputs/multiseed_budget_curve_20261104/) |
| [Table 4](Table_4.csv) | Catchment means and medians across metrics | [Source](../outputs/multimetric_transfer_20261102/transfer_endpoints.csv) |
| [Table 5](Table_5.csv) | Fixed-library deployment | [Source](../figures/Fig08/data/strategy_summary.csv) |
| [Table S2a](Table_S2a.csv) | Period, seed and catchment variance shares | [Source](../figures/Fig03/data/variance_shares.csv) |
| [Table S2b](Table_S2b.csv) | Late-period variance components | [Source](../outputs/training_domain_followup_20261102/unseen_aligned/reml.csv) |
| [Table S3](Table_S3.csv) | Single-instance resampling schemes | [Source](../outputs/jh_predraft_audit_20261102/global_baseline_interval_reconciliation.csv) |
| [Table S4a](Table_S4a.csv) | Budget endpoints and bounds | [Source](../figures/Fig05/data/curve_endpoints.csv) |
| [Table S4b](Table_S4b.csv) | Transfer ratios | [Source](../figures/Fig04/data/transfer_ratios.csv) |
| [Table S5a](Table_S5a.csv) | Catchment influence | [Source](../figures/Fig07/data/robustness_endpoints.csv) |
| [Table S5b](Table_S5b.csv) | Seven selection windows | [Source](../figures/Fig04/data/window_losses.csv) |
| [Table S6](Table_S6.csv) | Held-out and training comparisons | [Source](../figures/Fig07/data/matching_contrasts.csv) |
| [Table S7](Table_S7.csv) | Low-flow volume bias in 103 catchments | [Source](../outputs/metric_evidence_plan_20261102/flv_valid103_endpoints.csv) |
| [Table S8](Table_S8.csv) | Low-flow-targeted selection | [Source](../outputs/multimetric_transfer_20261102/transfer_endpoints.csv) |
| [Table S9a](Table_S9a.csv) | 370 training catchments | [Source](../outputs/multimetric_transfer_20261102/transfer_endpoints.csv) |
| [Table S9b](Table_S9b.csv) | External 256-unit archive | [Source](../outputs/weakness_resolution_20261102/external_NSE/selection_endpoints.csv) |
| [Table S10](Table_S10.csv) | Climate strata and split counts | [Source](../splits/formal_pub_527/climate_stratum_counts.csv) |

For Table S2b, the [training-catchment components](../outputs/weakness_resolution_20261102/local_internal/reml.csv) accompany the held-out estimates. For S3, the [seed-only estimates](../outputs/selection_sensitivity/estimates.json) accompany the two-way intervals. In the archived bootstrap matrices, row zero holds the observed point estimate. The percentile intervals use the following 2,000 rows. For S9b, the [MSE-trained external archive](../outputs/weakness_resolution_20261102/external_MSE/selection_endpoints.csv) accompanies the NSE-trained archive.

Table 3 and Table S4a use 20,000 HUC2 resamples conditional on the fitted library. At five source instances, the one-sided 95% upper bound on the transferred local increment is 0.0126 NSE for M1/B0 and 0.0550 NSE for EA/B0.
