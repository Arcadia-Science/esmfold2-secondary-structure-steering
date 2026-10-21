# Steering secondary structure in ESMFold2 binder design

This code repository contains or points to all materials required for creating and hosting the publication entitled, *"Steering secondary structure in ESMFold2 binder design"*.

The publication is hosted at [this URL](https://arcadia-science.github.io/2026-esmfold2-secondary-structure-steering/).

## Data Description

- `data/targets`: tracked target sequences and experimental coordinates.
- `data/campaigns`: campaign structures, model scores, traces and protocols. Not tracked.
- `data/reference`: frozen search results, entry metadata and RCSB coordinates. Not tracked.

The GPU outputs have not yet been publicly archived, so the untracked inputs must be
supplied locally. Analysis recalculates secondary structure, reference selection and
summaries from those inputs and never launches GPU work.

## Reproduce

Please see [SETUP.qmd](pages/SETUP.qmd).

## Contribute

Please see [CONTRIBUTING.qmd](pages/CONTRIBUTING.qmd).
