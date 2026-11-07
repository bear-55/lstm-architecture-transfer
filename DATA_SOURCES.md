# Data sources and attribution

## CAMELS-US

Time series: Newman, A., Sampson, K., Clark, M. P., Bock, A., Viger, R. J. and Blodgett, D. (2014), *A large-sample watershed-scale hydrometeorological dataset for the contiguous USA*, UCAR/NCAR. [DOI: 10.5065/D6MW2F4D](https://doi.org/10.5065/D6MW2F4D).

Attributes: Addor, N., Newman, A. J., Mizukami, N. and Clark, M. P. (2017), *Catchment attributes for large-sample studies*, UCAR/NCAR. [DOI: 10.5065/D6G73C3Q](https://doi.org/10.5065/D6G73C3Q).

The [UCAR catalogue record](https://data.ucar.edu/en/dataset/camels-catchment-attributes-and-meteorology-for-large-sample-studies) specifies **CC BY 4.0** for the CAMELS dataset. The repository includes selected catchment attributes and derived score tables; the training-cache attachment contains processed meteorological/attribute arrays and training/validation observations. Daily-prediction files include aligned observations, dates and model simulations. Processing includes catchment selection, unit conversion, training-fitted scaling and metric computation, as specified in the model/data code and cache metadata. Original full CAMELS download archives are obtained from the source provider.

Associated papers:

- Newman et al. (2015), *Development of a large-sample watershed-scale hydrometeorological data set for the contiguous USA*. [https://doi.org/10.5194/hess-19-209-2015](https://doi.org/10.5194/hess-19-209-2015).
- Addor et al. (2017), *The CAMELS data set: catchment attributes and meteorology for large-sample studies*. [https://doi.org/10.5194/hess-21-5293-2017](https://doi.org/10.5194/hess-21-5293-2017).

## External EA-LSTM comparison

Kratzert, F. (2021), *Pretrained models + simulations for “Towards learning universal, regional, and local hydrological behaviors via machine learning applied to large-sample datasets”*, HydroShare. [DOI: 10.4211/hs.83ea5312635e44dc824eeb99eda12f06](https://doi.org/10.4211/hs.83ea5312635e44dc824eeb99eda12f06).

The [resource](https://www.hydroshare.org/resource/83ea5312635e44dc824eeb99eda12f06/) is distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). This repository redistributes derived catchment/seed/period score arrays and plotting summaries. They are recalculated transformations of the authors' simulations. Source archive/run identifiers, checksums and the transformation code are included. The original model archive is retrieved from HydroShare by the optional download script.

Associated paper: Kratzert et al. (2019), *Towards learning universal, regional, and local hydrological behaviors via machine learning applied to large-sample datasets*. [https://doi.org/10.5194/hess-23-5089-2019](https://doi.org/10.5194/hess-23-5089-2019).

## Figure 2 map boundaries

State outlines use the US Census Bureau's ACS 2024 generalized state boundaries; HUC2 outlines use the USGS Watershed Boundary Dataset. Service URLs, downloaded geometries and source metadata are in `figures/Fig02/data/`. These US federal boundary products retain their agency attribution. The geometries are simplified for display; catchment locations and results are separate data layers.

## Licence scope

The MIT licence at repository root applies to the authors' code. It does not replace the above source-data licences. Third-party data and derived redistributions retain their attribution obligations. Cite the relevant original datasets and papers when reusing them. The frozen model library, experiment-specific results and figure code were produced by the authors of the present study.
