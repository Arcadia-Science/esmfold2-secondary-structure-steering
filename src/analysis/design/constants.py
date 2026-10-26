"""Vocabulary, loss weights, and defaults for ESMFold2 binder design."""

from __future__ import annotations

import torch
from esm.models.esmfold2 import ELEMENT_NUMBER_TO_SYMBOL
from esm.models.esmfold2.constants import PROTEIN_1TO3, RES_TYPE_TO_CCD

TrajectoryStep = dict[str, torch.Tensor | float]
Trajectory = dict[int, TrajectoryStep]


TOKENS = ["<pad>", "-"] + [RES_TYPE_TO_CCD[i] for i in range(2, 33)]
ELEMENTS = ["X"] * (max(ELEMENT_NUMBER_TO_SYMBOL) + 1)
ELEMENTS[0] = "<pad>"
for _atomic_num, _symbol in ELEMENT_NUMBER_TO_SYMBOL.items():
    ELEMENTS[_atomic_num] = _symbol[:1] + _symbol[1:].lower()
TOKEN_IDS = {token: idx for idx, token in enumerate(TOKENS)}
AA_DIMS = 20
CYS_IDX = TOKEN_IDS[PROTEIN_1TO3["C"]] - 2
MUTABLE_TOKEN = "#"
BinderPromptStr = str

LOSS_WEIGHTS = {"intra_contact": 0.5, "inter_contact": 0.5, "glob": 0.2}
STEPS = 150
LOG_INTERVAL = 5
LEARNING_RATE = 0.1
TEMPERATURE_MIN = 1e-2
ESMC_MASK_FRACTION = 0.15
LM_LOSS_BATCH_SIZE = 128
LM_MASK_PASSES = 4
BYTES_PER_GIB = 1024**3
COMPILE = True
REUSE_ESMC = True
