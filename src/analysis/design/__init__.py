"""ESMFold2 binder design, ported from the ESM cookbook.

This subpackage is a refactor of the ESM cookbook binder-design tutorial,
``cookbook/tutorials/binder_design.py`` in https://github.com/Biohub/esm at
commit ``ba4d712``. The upstream file is one 1322-line script; here it is split
along its own section boundaries into modules, with the code moved verbatim.
Nothing was renamed and no numerical behaviour was altered. See ``NOTICE`` at
the repository root for the attribution, and the module list below for the map
back to upstream line numbers.

One definition is extended rather than reproduced: ``design_binder`` in
:mod:`analysis.design.loop`, which gains the secondary-structure steering terms,
the per-step binding readout, and the per-step loss log that this publication is
about. Four upstream statements change there and the rest is inserted; with the
default schedule the objective is still exactly the cookbook's, because the
steering total is then a zero with no gradient behind it.
``tests/test_design_port.py`` enumerates that divergence and fails if it grows.

Upstream is MIT licensed:

    Copyright 2026 Chan Zuckerberg Biohub, Inc.

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.

Layout, with the upstream line ranges each module holds:

- :mod:`analysis.design.constants` (69-70, 77-105)
- :mod:`analysis.design.prompts` (111-127, 131-151, 154-165)
- :mod:`analysis.design.sequence` (172-216)
- :mod:`analysis.design.losses` (219-254, 260-363)
- :mod:`analysis.design.metrics` (369-473)
- :mod:`analysis.design.folding` (479-596)
- :mod:`analysis.design.structure` (599-682)
- :mod:`analysis.design.plm` (688-809)
- :mod:`analysis.design.models` (1061-1095)
- :mod:`analysis.design.loop` (64-67, 815-825, 828-1056, 1098-1176), extended

:mod:`analysis.design.initialization` adds this publication's matched-input
experiments on top of the shared engine; it has no upstream counterpart.

Importing this package needs ``esm`` and the Biohub ``transformers`` fork.
Running a design also needs a CUDA device: the ported code calls ``.cuda()``
directly, as upstream does.
"""

from __future__ import annotations

from analysis.design.constants import (
    LOSS_WEIGHTS,
    STEPS,
    Trajectory,
    TrajectoryStep,
)
from analysis.design.loop import ESMFold2Design, design_binder, normalized_gradient_tensor
from analysis.design.losses import compute_structure_losses
from analysis.design.metrics import compute_distogram_iptm_proxy
from analysis.design.prompts import (
    BINDER_PROMPT_FACTORIES,
    TARGET_SEQUENCES,
    PromptFactory,
)

__all__ = [
    "BINDER_PROMPT_FACTORIES",
    "ESMFold2Design",
    "LOSS_WEIGHTS",
    "STEPS",
    "TARGET_SEQUENCES",
    "PromptFactory",
    "Trajectory",
    "TrajectoryStep",
    "compute_distogram_iptm_proxy",
    "compute_structure_losses",
    "design_binder",
    "normalized_gradient_tensor",
]
