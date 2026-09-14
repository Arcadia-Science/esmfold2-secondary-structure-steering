"""Analysis code for steering secondary structure in ESMFold2 binder design.

Layout:

- :mod:`analysis.paths` resolves every artifact location from the repo root.
- :mod:`analysis.targets` holds the design targets and the loss arms.
- :mod:`analysis.steering` holds the secondary-structure loss terms themselves,
  which need torch but neither ``esm`` nor a GPU, so the shape of a term is
  something the notebook can check on its own.
- :mod:`analysis.schedule` holds the per-step weight schedules that turn a term
  into an arm.
- :mod:`analysis.loss_trace` reads and writes the per-step loss logs.
- :mod:`analysis.deposit` fetches the deposited inputs, and holds the one
  switch that regenerates them from source instead.
- :mod:`analysis.confidence` splits ESMFold2's per-residue pLDDT by chain, so
  binder confidence can be read separately from the complex it sits in.
- :mod:`analysis.artifacts` is the file-exists gate the notebook reads through.
- :mod:`analysis.composition` holds the per-design composition tables.
- :mod:`analysis.reference` loads the natural binder baseline, fetching it
  from the deposit when absent.
- :mod:`analysis.secondary_structure` measures composition, one way everywhere.
- :mod:`analysis.design` is the design loop itself, a refactor of the ESM
  cookbook binder-design tutorial. It needs ``esm`` and a CUDA device, so it is
  imported only inside the GPU container and is not one of this package's eager
  imports.
- :mod:`analysis.modal_app` is the GPU app, deployable by a reader.
- :mod:`analysis.campaign` launches and collects campaigns, skipping any that
  already have results.

Executing the notebook only ever touches the modules above
:mod:`analysis.design`. Campaign generation is
a separate, explicit act that costs GPU time and will not reproduce the exact
designs in the figures, because the design loop is not seed-reproducible.
"""

from __future__ import annotations

from analysis import (
    artifacts,
    composition,
    confidence,
    deposit,
    loss_trace,
    paths,
    reference,
    schedule,
    secondary_structure,
    steering,
    targets,
)

__all__ = [
    "artifacts",
    "composition",
    "confidence",
    "deposit",
    "loss_trace",
    "paths",
    "reference",
    "schedule",
    "secondary_structure",
    "steering",
    "targets",
]
