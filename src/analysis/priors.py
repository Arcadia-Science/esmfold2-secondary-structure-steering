"""Empirical three-state DSSP priors and matched soft sequence initialization."""

from __future__ import annotations
from pathlib import Path

import numpy as np
from biotite.sequence import ProteinSequence

from analysis import paths, rcsb, reference
from analysis import secondary_structure as ss
from analysis.parallel import DEFAULT_WORKERS, starmap, validate_workers
from analysis.targets import INITIALIZATION_SPECS

AA = list("ADEFGHIKLMNPQRSTVWY")


def tempered_sheet(prior: dict, temperature: int) -> tuple[np.ndarray, float]:
    """Flatten the prior and scale the identity-preserving boost with its range."""
    if temperature not in (1, 2, 4):
        raise ValueError("The published temperatures are 1, 2, and 4")
    q = np.asarray(prior["sheet"], dtype=float)
    if prior["amino_acid_order"] != AA or q.shape != (19,) or not np.all(q > 0):
        raise ValueError("Expected a positive 19-amino-acid prior excluding cysteine")
    if not np.isclose(q.sum(), 1) or np.ptp(q) == 0:
        raise ValueError("Expected a normalized, nonuniform sheet prior")
    qt = np.exp(np.log(q) / temperature)
    qt /= qt.sum()
    epsilon = INITIALIZATION_SPECS["short"]["epsilon"]
    boost = epsilon if temperature == 1 else float(epsilon * np.ptp(qt) / np.ptp(q))
    return qt, boost


def matched_probabilities(q: np.ndarray, sequence: str, boost: float) -> np.ndarray:
    """Preserve the chosen hard identity at every position without freezing gradients."""
    q = np.asarray(q)
    chosen = np.asarray([AA.index(aa) for aa in sequence])
    p = q + boost * (np.eye(19)[chosen] - 1 / 19)
    if not (p > 0).all() or not np.allclose(p.sum(-1), 1):
        raise ValueError("The boost does not give positive normalized probabilities")
    if not np.array_equal(p.argmax(-1), chosen):
        raise ValueError("The boost does not preserve the requested hard sequence")
    return p


def paired_inputs() -> list[dict]:
    """The five fixed-length matched inputs used for the short trajectories."""
    bundles = []
    settings = INITIALIZATION_SPECS["short"]
    for pair in range(settings["pairs"]):
        chosen = np.random.default_rng(2026092500 + pair).integers(
            0, 19, size=settings["binder_length"]
        )
        sequence = "".join(AA[i] for i in chosen)
        bundle = {"pair": pair, "sequence": sequence, "aa_order": AA}
        bundles.append(bundle)
    return bundles


def count_entry(cif_path: Path, chains: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Count eligible residues from one PDB; importable by spawned workers."""
    counts = {state: np.zeros(len(AA), dtype=np.int64) for state in ("helix", "sheet")}
    protein = rcsb._protein(cif_path)
    for chain in chains:
        backbone = ss.backbone_residues(protein[protein.chain_id == chain])
        labels = ss.assign(backbone)
        for label, state in (("H", "helix"), ("E", "sheet")):
            for name in backbone.res_names[labels == label]:
                try:
                    aa = ProteinSequence.convert_letter_3to1(name)
                except KeyError:
                    continue
                # Modified residues must not be mapped into canonical counts.
                if aa in AA and ProteinSequence.convert_letter_1to3(aa) == name:
                    counts[state][AA.index(aa)] += 1
    return counts


def from_reference(natural=None, *, workers: int = DEFAULT_WORKERS) -> dict:
    """Rebuild pooled priors from the recorded natural reference and cached RCSB CIFs.

    All H assignments enter helix, including 3-10 and pi helix as classified
    by the shared three-state PyDSSP implementation. Noncanonical residues
    and cysteine are excluded; original chains are counted once, without resampling.
    Each spawned worker counts one PDB. Integer counts are summed before
    normalizing, so process scheduling cannot affect the probabilities.
    ``workers=1`` runs serially, including reference selection when needed.
    """
    validate_workers(workers)
    if natural is None:
        natural = reference.composition(workers=workers)
    natural = natural[natural.in_reference_set == 1]
    jobs = [
        (paths.REFERENCE / "cifs" / f"{pdb_id}.cif.gz", tuple(members.binder_auth_chain))
        for pdb_id, members in natural.groupby("pdb_id", sort=True)
    ]
    counts = {state: np.zeros(len(AA), dtype=np.int64) for state in ("helix", "sheet")}
    for entry_counts in starmap(count_entry, jobs, workers=workers):
        for state in counts:
            counts[state] += entry_counts[state]
    return {
        "amino_acid_order": AA,
        "excluded_amino_acid": "C",
        **{state: (count / count.sum()).tolist() for state, count in counts.items()},
        "residue_counts": {state: int(count.sum()) for state, count in counts.items()},
        "uniform": [1 / 19] * 19,
        "definition": "P(amino acid | DSSP state, canonical and non-Cys)",
    }
