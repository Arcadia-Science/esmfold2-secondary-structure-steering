# Steering secondary structure in ESMFold2 binder design

This code repository contains or points to all materials required for creating and hosting the publication entitled, *"Steering secondary structure in ESMFold2 binder design"*.

The publication is hosted at [this URL](https://arcadia-science.github.io/esmfold2-secondary-structure-steering/).

## Data Description

- `data/targets`: tracked target sequences and experimental coordinates.
- `data/campaigns`: campaign structures, model scores, traces and protocols. Not tracked.
- `data/reference`: frozen search results, entry metadata and RCSB coordinates. Not tracked.

Campaign outputs and frozen natural-reference inputs are archived on
[Zenodo](https://doi.org/10.5281/zenodo.23542792) and downloaded automatically when
needed. Analysis recalculates secondary structure, reference selection and summaries
from those inputs.

## Reproduce

Please see [SETUP.qmd](pages/SETUP.qmd).

## Contribute

Please see [CONTRIBUTING.qmd](pages/CONTRIBUTING.qmd).
