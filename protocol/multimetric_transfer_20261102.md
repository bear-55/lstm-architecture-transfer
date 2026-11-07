# Multi-metric selection and retraining transfer

This supplementary analysis plan was written on 2026-11-02, before computing the multi-metric D values and after inspecting the NSE/P1 and fixed-library results.

## Scope

Use the existing predictions from ten instances each of B0, M1 and EA in 107 held-out and 370 training catchments. Select during 1999-10-01 to 2003-09-30 and evaluate during 2003-10-01 to 2008-09-30. Use early catchment-wise NSE for the main selection rule and score those choices on multiple late-period metrics. Report selection based on each metric's own early values separately. The plan uses the existing weights.

## Metrics and utility

1. Higher NSE and NSE* = NSE/(2 − NSE) are better. Bounded NSE checks sensitivity to the score scale.
2. Higher KGE (2009) is better. Its correlation, variability and mean-ratio components describe different aspects of hydrograph fit.
3. FHV is volume bias on the top 2% of the flow-duration curve. Report signed and absolute FHV and use −|FHV| as utility in percentage points.
4. Compute MAE in mm/day below observed Q20 and above Q95 within each period. Utility is negative MAE.
5. Normalize each MAE by mean daily observed flow in the same catchment and period. Report both normalized and raw errors, listing non-positive or non-finite denominators.
6. Report signed volume bias above observed Q98 as a supplement, with negative absolute bias as utility.
7. Near-zero denominators can produce extreme FLV values. This initial plan retains catchment values and the median and excludes regional FLV means and D. The separately documented valid-subset sensitivity reports subsequent FLV analyses.

Report all listed metrics as complementary diagnostics of the same predictions.

## Estimation

For each source seed, choose M1 or B0, or EA or B0, using the early metric. Score the choice with the source weights and with each of the other nine evaluation seeds. Measure gains relative to the corresponding B0 instance. Define D = G_same − G_new.

Use 2,000 synchronized seed/HUC2 percentile bootstrap draws in the P1 sequence with RNG seed 20261102. Exclude matching source and evaluation seed identities even if resampling duplicates a seed. Report regional means and medians of catchment-level seed means, with pointwise 95% intervals.

The global rule selects using the early metric averaged over catchments for each source seed. Use NSE for primary selection and the corresponding utility for metric-specific selection. Report the global and local components. The additive decomposition applies to means.

For the fixed 20-member library, use the established early NSE choices for local instance selection, architecture choice followed by ensembling, pooled and single-architecture ensembles and global instance references. Include all catchments and report signed and absolute biases, MAE and KGE components. Figures summarize catchment medians, distinguishing typical signed bias from typical absolute bias.

## Checks

Reproduce P1 point estimates and intervals for NSE and NSE*. Verify the matrix calculation by direct enumeration of the 10 × 9 source/evaluation pairs. Verify absolute biases in selection, mean discharge and prediction dates. Record input SHA256 values and code with the CSV and figure outputs.
