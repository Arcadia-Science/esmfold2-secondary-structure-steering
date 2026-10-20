# Steering secondary structure in ESMFold2 binder design

This repository contains the publication notebook, analysis code, target inputs,
and GPU experiment runners for secondary-structure steering in ESMFold2 binder design.

## Reproduce the analysis

Create the environment from `env.yml`, activate it, install this package with
`pip install -e .`, and run `make execute`. See the [setup guide](pages/SETUP.qmd)
for installation instructions. Input requirements are listed below.

GPU outputs have not yet been publicly archived. Until they are available,
execution requires the original local model outputs. Missing inputs produce an
error; executing the publication never submits GPU jobs. Rerunning GPU generation
produces new samples and does not reproduce the exact reported designs.

Analysis recalculates secondary structure, reference selection,
and summaries from preserved inputs. Model data and exploratory work remain outside
source control. Executed notebooks and `_freeze/` results are retained as required
by the publication template, so publishing does not rerun the analysis.

## Code and inputs

- `src/analysis`: shared analysis and generation code; experimental settings are in
  `analysis.targets`.
- `data/targets`: tracked target sequences and experimental coordinates.
- `data/campaigns`: ignored campaign structures, model scores, traces and protocols.
- `data/reference`: ignored frozen search results, entry metadata and RCSB coordinates.

## Generate model outputs

Use the project environment and a configured Modal account with access to the
ESMFold2/ESMC models. Deploy the single shared app:

```bash
modal deploy -m analysis.modal_app
```

Unsteered, steered, and onset/release campaigns share the target and arm registry:

```bash
python -m analysis.campaign spawn --target pdl1_igv --arm vanilla
python -m analysis.campaign collect --target pdl1_igv --arm vanilla
```

Package preserved inputs locally, without uploading, using:

```bash
python -m analysis.campaign package --experiment campaigns
python -m analysis.campaign package --experiment reference
```

Reference measurements use up to four spawned CPU processes by default.
Use `reference.composition(workers=1)` for serial execution.

Sequence-initialization experiments now live in the companion repository,
[Initialization-conditioned generation with ESMFold2](https://github.com/Arcadia-Science/2026-esmfold2-initialization-conditioned-generation).
This repository retains the five-target loss-steering grid, the entropy/helicity
analysis, and the PD-L1 timing appendix. Each publication carries its own snapshot
of the shared analysis and design engine.

See the [publishing guide](developer-docs/PUBLISHING_GUIDE.md) for the template's
tagged-version build and publication process.
