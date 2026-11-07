# External replication and follow-up analyses

This plan was written after examining the local results and before accessing the external daily simulations. The local follow-up analyses are exploratory. The freeze record gives the local timestamp and original SHA256. There was no third-party registration.

## Data and contrasts

Use the HydroShare archive with DOI 10.4211/hs.83ea5312635e44dc824eeb99eda12f06. Verify its inventory, configurations, training and test periods, static inputs and seed identifiers. Compare the standard concatenation LSTM with the EA-LSTM using matched forcing, static information and loss. Give priority to the authors' basin-normalized NSE loss. If an exact pair is unavailable, describe the closest available pair and its differences. Keep all complete and readable runs without screening on test performance. Record missing or failed runs. Other losses, attribute sets and conceptual models are secondary comparisons, with no presumed ordering of effects.

## Periods and decisions

Use simulations withheld from fitting and validation-based model selection. Establish this status from the metadata before treating a contrast as confirmatory. Divide the test period into complete October–September water years. Use the first floor(Y/2) years for selection and the remaining years for evaluation. Require two complete years per period. Shorter records support only a single-period variance analysis. Within each catchment and period, use dates valid for all configurations, treat documented negative-discharge sentinels as missing and retain valid low scores. Record catchment and day counts. Use selection-period observations for all choices. Resolve NSE ties lexicographically by model name and run ID.

## Endpoints and uncertainty

1. The primary endpoint is the late-period decline in mean NSE gain, D, when historical catchment choices are transferred from retained weights to independently trained instances. Report a positive lower 95% bound as directional replication and report 0.01 separately as a magnitude reference.
2. Pair seeds across architectures only if the archive confirms corresponding IDs. Otherwise enumerate all source-run pairs and exclude each source run from its architecture's evaluation pool. Report the variance decomposition for a lexicographic pairing and sensitivity across all pairings.
3. Fit a crossed catchment and seed random-effects model for score differences with non-negative REML variance components and an independent Gaussian working error. The residual combines the interaction and cell error in the unreplicated grid. Report sums of squares and REML. A residual share above 50% is a descriptive threshold.
4. Compare local and global selection, within-architecture seed ensembles, pooled ensembles and architecture selection followed by ensembling. Distinguish retained weights from an independently trained pool. Average daily predictions before scoring ensembles. Match training budgets and report deployed ensemble sizes.
5. Use 2,000 crossed seed/catchment bootstrap draws with RNG seed 20261102. If HUC2 is available, also resample seeds and whole HUC2 regions. Keep complete seed rows, catchment or region columns, and both periods. Repeat selection in the historical period for each draw. Subsets overlap.
6. Use raw NSE as the main scale. Check bounded NSE* = NSE/(2 − NSE), within-seed catchment ranks of score differences, and KGE (2009). Planned sensitivities retain catchments with non-negative early NSE for all candidates or remove the five catchments with the largest early absolute mean difference. Identify additional screens based on later scores separately.
7. Report a descriptive period × seed × catchment sums-of-squares decomposition for the two fixed periods.

## Seed budget and flow diagnostics

Compute rho(n) = v_b/(v_b + v_e/n) from the REML components for n = 1–50. This is the model-based reliability of relative catchment differences for the observed period and configuration. Show reference levels 0.5 and 0.8 and distinguish extrapolation beyond the observed seed count. At the zero boundary for v_b, a finite positive-reliability planning estimate is unavailable. Also report correlations and selection gains from observed splits or independent pools.

Use KGE (2009) on raw discharge, with correlation r, variability ratio alpha and mean ratio beta. FHV measures relative volume bias in the top 2% of separately ranked flow-duration curves. FLV uses a logarithmic volume definition over the lowest 30%, documenting zero handling in the reference implementation. Also report MAE on days selected by observed Q95 and Q20 and signed error on observed peak days.

## Internal extension and reporting

Use the 30 selected checkpoints for inference on the 370 training catchments in 1999-10-01 to 2003-09-30 and 2003-10-01 to 2008-09-30. Report this temporal extension separately from the 107 held-out catchments. Record the analysis chronology. M1 is a small architectural change. Compare architectures under a common hyperparameter protocol, document parameter counts and implementations, and recognize that tuning may affect both average performance and training variation. Assemble configurations, checkpoints, predictions, scripts, source attribution, licensing, environment details and a figure entry point before repository or DOI release.
