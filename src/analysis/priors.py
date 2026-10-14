"""Empirical three-state DSSP priors and matched soft sequence initialization."""

from __future__ import annotations
from pathlib import Path

import numpy as np
from biotite.sequence import ProteinSequence

from analysis import paths, rcsb, reference
from analysis import secondary_structure as ss
from analysis.parallel import DEFAULT_WORKERS, starmap, validate_workers
from analysis.targets import INITIALIZATION_MARGIN, INITIALIZATION_SPECS

AA = list("ADEFGHIKLMNPQRSTVWY")


def _probability_vector(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    if q.shape != (len(AA),) or not np.isfinite(q).all() or not (q > 0).all():
        raise ValueError("Expected 19 positive finite probabilities excluding cysteine")
    if not np.isclose(q.sum(), 1, atol=1e-12, rtol=1e-9):
        raise ValueError("Expected a normalized amino-acid prior")
    return q


def validate_temperature(temperature: float) -> None:
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Initialization temperature must be positive and finite")


def tempered_prior(q: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Return normalized q**(1/temperature); values above one flatten the prior."""
    validate_temperature(temperature)
    q = _probability_vector(q)
    if temperature == 1:
        return q.copy()
    scores = np.log(q)
    with np.errstate(over="ignore", under="ignore"):
        qt = np.exp((scores - scores.max()) / temperature)
    qt /= qt.sum()
    if not (qt > 0).all():
        raise ValueError("Initialization temperature produces nonpositive prior probabilities")
    return qt


def minimum_boosts(
    q: np.ndarray,
    sequence: str,
    *,
    temperature: float = 1.0,
    margin: float = INITIALIZATION_MARGIN,
) -> np.ndarray:
    """Smallest nonnegative boosts that give each chosen identity the required lead.

    Temper the prior first, then calculate independently for each position.
    An identity that already exceeds all competitors by ``margin`` needs no
    adjustment. The margin is measured in the final probabilities.
    """
    return _minimum_boosts(tempered_prior(q, temperature), sequence, margin)


def _minimum_boosts(q: np.ndarray, sequence: str, margin: float) -> np.ndarray:
    if not np.isfinite(margin) or margin <= 0:
        raise ValueError("Probability margin must be positive and finite")
    if not sequence or any(aa not in AA for aa in sequence):
        raise ValueError("Expected a nonempty sequence of canonical, non-cysteine amino acids")
    chosen = np.asarray([AA.index(aa) for aa in sequence])
    competitors = np.broadcast_to(q, (len(sequence), len(AA))).copy()
    competitors[np.arange(len(sequence)), chosen] = -np.inf
    return np.maximum(0, competitors.max(axis=1) - q[chosen] + margin)


def matched_probabilities(
    q: np.ndarray,
    sequence: str,
    *,
    temperature: float = 1.0,
    margin: float = INITIALIZATION_MARGIN,
) -> np.ndarray:
    """Preserve the hard sequence with each prior's minimum position-specific boost.

    Temper first: q_T(a) is proportional to q(a)**(1/temperature).
    Then p_i(a) = q_T(a) + b_i * (1[a = a_i] - 1/19), keeping the sum at one.
    Reject priors/margins that make any probability nonpositive rather than
    clipping or renormalizing, which would change the intended margin.
    """
    q = tempered_prior(q, temperature)
    boosts = _minimum_boosts(q, sequence, margin)
    chosen = np.asarray([AA.index(aa) for aa in sequence])
    p = q + boosts[:, None] * (np.eye(len(AA))[chosen] - 1 / len(AA))
    if not (p > 0).all() or not np.isfinite(p).all():
        raise ValueError("Minimum boost would make probabilities nonpositive for this prior/margin")
    if not np.array_equal(p.argmax(-1), chosen):
        raise ValueError("Probability margin is too small to preserve the requested hard sequence")
    return p


def validate_initialization_settings(protocol: dict) -> None:
    """Require a positive finite temperature and probability margin."""
    validate_temperature(protocol.get("initialization_temperature", 0))
    margin = protocol.get("probability_margin", 0)
    if not np.isfinite(margin) or margin <= 0:
        raise ValueError("Probability margin must be positive and finite")


def paired_inputs() -> list[dict]:
    """Fixed-length starting sequences shared by every short-trajectory arm."""
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
    counts = count_reference(natural, workers=workers)
    return {
        "amino_acid_order": AA,
        "excluded_amino_acid": "C",
        **{state: (count / count.sum()).tolist() for state, count in counts.items()},
        "residue_counts": {state: int(count.sum()) for state, count in counts.items()},
        "uniform": [1 / 19] * 19,
        "definition": "P(amino acid | DSSP state, canonical and non-Cys)",
    }


def count_reference(natural=None, *, workers: int = DEFAULT_WORKERS) -> dict[str, np.ndarray]:
    """Pool canonical non-cysteine residue counts by DSSP state across reference chains."""
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
    return counts
