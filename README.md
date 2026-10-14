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

Analysis recalculates secondary structure, reference selection, amino-acid priors,
and summaries from preserved inputs. Model data and exploratory work remain outside
source control. Executed notebooks and `_freeze/` results are retained as required
by the publication template, so publishing does not rerun the analysis.

## Code and inputs

- `src/analysis`: shared analysis and generation code; experimental settings are in
  `analysis.targets`.
- `data/targets`: tracked target sequences and experimental coordinates.
- `data/campaigns`: ignored campaign structures, model scores, traces and protocols.
- `data/reference`: ignored frozen search results, entry metadata and RCSB coordinates.
- `data/initialization`: ignored original initialization runs, including protocols,
  exact input tensors, model outputs and execution provenance.

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

Initialization experiments use the same campaign CLI and deployment. They first
prepare matched inputs. `softness` performs fixed-input forwards, `short` runs
40-update matched uniform/helix/sheet trajectories, and `full` runs 150-update designs.
The short experiment uses 10 shared starting sequences of length 120: each job
runs all three arms with identical initial atom features, for 30 trajectories.
Its uniform control uses the same minimum-boost rule as the empirical priors,
rather than the default initializer's random score perturbations.

```bash
python -m analysis.campaign prepare --experiment short
python -m analysis.campaign spawn --experiment short
python -m analysis.campaign collect --experiment short

python -m analysis.campaign prepare --experiment full --temperature 1
python -m analysis.campaign spawn --experiment full --temperature 1
python -m analysis.campaign collect --experiment full --temperature 1
```

Use `--experiment softness` for the zero-update screen. Its preparation runs a
Modal CPU function; other preparation is local. Full runs require the preserved
unsteered campaign's initial hard sequences. Collect every full T=1 run before
preparing other full-run temperatures: they reuse the exact T=1 atom-feature
tensors. Both `short` and `full` accept a positive finite `--temperature`,
including fractional values such as `1.5`. Pass the same value to prepare,
spawn, and collect. The publication's full-run grid remains T=1, 2, and 4.

Use an absolute `--root` for each initialization study and pass it consistently
to prepare, spawn, and collect. Preparation refuses to overwrite an existing
protocol. Jobs use the shared `manifest.csv` format.

Short and full initialization runs first temper each prior as
`q_T = normalize(q ** (1 / temperature))`. Temperature defaults to 1 (the
original prior); values above 1 flatten it, and values below 1 sharpen it.
With probability margin `delta = 0.001`, the smallest nonnegative boost at each
position is `max(0, max_other(q_T) - q_T(chosen) + delta)`. The chosen probability
gains `18 * boost / 19`; every other allowed amino acid loses `boost / 19`.
A chosen identity that already leads by the margin leaves the tempered prior
unchanged. Cysteine stays excluded. Prior/temperature combinations that would
produce nonpositive probabilities are rejected rather than clipped or renormalized.

The helix and sheet priors get independently calculated boosts while retaining
the same initial discrete sequences. Temperature is applied before calculating
boosts, so the margin remains an absolute probability difference. Both workers
use the same probability and float32 logit-conversion code. Initialization
temperature and probability margin are recorded with the experiment settings;
the design loop's optimization temperature schedule remains independent.

The Python initializer accepts the same parameter:

```python
from analysis import priors

probabilities = priors.matched_probabilities(q, sequence, temperature=1.5)
```

The standard design worker handles loss-steered designs and prior-initialized
trajectories. The fixed-input screen uses a separate worker class within the same
app because it needs one uncompiled model. Experimental settings remain in
`analysis.targets`; deployment and model loading remain in `analysis.modal_app`.

Package preserved inputs locally, without uploading, using:

```bash
python -m analysis.campaign package --experiment campaigns
python -m analysis.campaign package --experiment initialization
python -m analysis.campaign package --experiment reference
```

Reference measurements and amino acid priors use up to four spawned CPU processes
by default. From a notebook, use `reference.composition(workers=1)` or
`priors.from_reference(natural, workers=1)` for serial execution. Worker functions
are importable modules, so notebook cells need no multiprocessing entry-point
guard. Ordinary scripts must call them inside `if __name__ == "__main__":`.
`python -m analysis.campaign prior` prints freshly calculated priors without
saving a measurement cache.

See the [publishing guide](developer-docs/PUBLISHING_GUIDE.md) for the template's
tagged-version build and publication process.
