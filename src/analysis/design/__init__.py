"""ESMFold2 binder design adapted from the ESM cookbook at commit ba4d712.

The local loop adds steering and diagnostics. See NOTICE for attribution.
Imports require ESM and the Biohub transformers fork; design execution uses CUDA.

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
