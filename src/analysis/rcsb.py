"""Measure the natural reference from frozen search hits, entry metadata and CIFs.

Inputs are retained locally and archived. A live search is needed only to create
new inputs; its release-date cutoff alone cannot freeze RCSB cluster membership.
DSSP, interfaces, exclusions and summaries are recalculated on every build.
"""

from __future__ import annotations
import gzip
import io
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import numpy as np
import pandas as pd

from analysis import paths, secondary_structure
from analysis.parallel import DEFAULT_WORKERS, starmap, validate_workers

SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
GRAPHQL_URL = "https://data.rcsb.org/graphql"
FILE_URL = "https://files.rcsb.org/download/{pdb_id}.cif.gz"

SNAPSHOT_RELEASE_DATE = "2026-09-02"
"""Initial-release ceiling for the query.

Set to the newest release in the deposited table, so re-running reproduces the
same candidate entries rather than picking up everything released since.
Raising it is a deliberate act that changes the reference set.
"""

MIN_LENGTH_RESIDUES = 60
MAX_LENGTH_RESIDUES = 200
MAX_RESOLUTION_ANGSTROM = 2.5
IDENTITY_CUTOFF_PERCENT = 30
MIN_OBSERVED_FRACTION = 0.80
MIN_TARGET_LENGTH_RESIDUES = 30
MIN_PLAUSIBLE_BSA_ANGSTROM2 = 800.0
CONTACT_CUTOFF_ANGSTROM = 5.0
SASA_POINT_NUMBER = 500

GRAPHQL_BATCH = 80
POST_ATTEMPTS = 5
DOWNLOAD_ATTEMPTS = 4
DOWNLOAD_WORKERS = 8

DESIGN_KEYWORDS = (
    "de novo",
    "designed protein",
    "computational design",
    "synthetic construct",
)
ENGINEERED_SCAFFOLD_KEYWORDS = (
    "nanobody",
    "antibody",
    "monobody",
    "affibody",
    "darpin",
    "anticalin",
    "minibinder",
    "single-domain",
    "single domain",
    "vhh",
    "scfv",
    "fab ",
    "immunoglobulin",
    "heavy chain variable",
    "light chain variable",
)

CIF_CACHE = paths.REFERENCE / "cifs"
SEARCH_HITS = "natural_binder_search_hits.json"
"""Frozen search results.

Recording the result sets, rather than only the query that produced them, is
what pins the set. The query is already capped by release date, but the RCSB
recomputes its sequence-identity clusters over the whole present-day PDB, so
which entity represents a cluster drifts under a live query and does not under
a recorded one. When this file is present the build reads it; when it is
absent the build queries the live API and writes it.
"""


@dataclass(frozen=True)
class EntityMeta:
    """One polymer entity of one entry, as the RCSB describes it."""

    entity_id: str
    pdb_id: str
    auth_chain: str
    n_instances: int
    sample_length_residues: int
    description: str
    organism: str
    taxonomy_id: int
    cluster_30: str


@dataclass(frozen=True)
class EntryMeta:
    """One PDB entry and its two polymer entities."""

    pdb_id: str
    title: str
    resolution_angstrom: float
    deposit_date: str
    release_date: str
    deposited_instances: int
    entities: tuple[EntityMeta, ...]


@dataclass
class EntryResult:
    """Measurements for one entry, plus whatever failed in it."""

    rows: list[dict[str, object]] = field(default_factory=list)
    failures: dict[str, str] = field(default_factory=dict)


def search_body(grouped: bool) -> dict[str, object]:
    """The RCSB Search API request.

    ``grouped`` switches between every in-range entity and one representative
    per 30% sequence-identity cluster. Both are needed: the representatives are
    what gets measured, and the full list is what says whether an entity's
    partner is also in range.
    """

    def term(
        attribute: str, operator: str, value: object, negation: bool = False
    ) -> dict[str, object]:
        parameters: dict[str, object] = {"attribute": attribute, "operator": operator}
        if value is not None:
            parameters["value"] = value
        if negation:
            parameters["negation"] = True
        return {"type": "terminal", "service": "text", "parameters": parameters}

    nodes = [
        term("exptl.method", "exact_match", "X-RAY DIFFRACTION"),
        term(
            "rcsb_entry_info.resolution_combined",
            "less_or_equal",
            MAX_RESOLUTION_ANGSTROM,
        ),
        term("rcsb_entry_info.polymer_entity_count", "equals", 2),
        term(
            "rcsb_entry_info.polymer_composition",
            "exact_match",
            "heteromeric protein",
        ),
        term(
            "entity_poly.rcsb_sample_sequence_length",
            "range",
            {
                "from": MIN_LENGTH_RESIDUES,
                "to": MAX_LENGTH_RESIDUES,
                "include_lower": True,
                "include_upper": True,
            },
        ),
        term("rcsb_entity_source_organism.ncbi_scientific_name", "exists", None),
        term(
            "rcsb_entity_source_organism.ncbi_scientific_name",
            "exact_match",
            "synthetic construct",
            negation=True,
        ),
        term(
            "rcsb_accession_info.initial_release_date",
            "less_or_equal",
            SNAPSHOT_RELEASE_DATE,
        ),
    ]
    request_options: dict[str, object] = {
        "return_all_hits": True,
        "results_content_type": ["experimental"],
    }
    if grouped:
        request_options["group_by"] = {
            "aggregation_method": "sequence_identity",
            "similarity_cutoff": IDENTITY_CUTOFF_PERCENT,
        }
        request_options["group_by_return_type"] = "representatives"
    return {
        "query": {"type": "group", "logical_operator": "and", "nodes": nodes},
        "return_type": "polymer_entity",
        "request_options": request_options,
    }


def post_json(url: str, body: dict[str, object]) -> dict[str, object]:
    """POST JSON with retries, because these are public endpoints under load."""
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    last = ""
    for attempt in range(POST_ATTEMPTS):
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                return json.loads(response.read())
        except Exception as error:  # noqa: BLE001
            last = str(error)
            time.sleep(3.0 * (attempt + 1))
    raise RuntimeError(f"POST to {url} failed after {POST_ATTEMPTS} attempts: {last}")


GRAPHQL_TEMPLATE = """
query ($ids: [String!]!) {
  entries(entry_ids: $ids) {
    rcsb_id
    struct { title }
    rcsb_entry_info {
      resolution_combined
      deposited_polymer_entity_instance_count
    }
    rcsb_accession_info { deposit_date initial_release_date }
    polymer_entities {
      rcsb_id
      entity_poly { rcsb_sample_sequence_length }
      rcsb_polymer_entity { pdbx_description }
      rcsb_polymer_entity_container_identifiers { auth_asym_ids }
      rcsb_entity_source_organism { ncbi_scientific_name ncbi_taxonomy_id }
      rcsb_polymer_entity_group_membership {
        group_id
        aggregation_method
        similarity_cutoff
      }
    }
  }
}
"""


def cluster_30(memberships: list[dict[str, object]] | None) -> str:
    """The entity's 30% sequence-identity cluster id, or empty if it has none."""
    if not memberships:
        return ""
    for membership in memberships:
        method = str(membership.get("aggregation_method") or "")
        cutoff = membership.get("similarity_cutoff")
        if method == "sequence_identity" and cutoff is not None:
            if int(float(cutoff)) == IDENTITY_CUTOFF_PERCENT:
                return str(membership["group_id"])
    return ""


def parse_entity(pdb_id: str, entity: dict[str, object]) -> EntityMeta:
    """One GraphQL polymer entity as an :class:`EntityMeta`.

    ``auth_chain`` takes the first author chain id in sorted order. An entity
    present as several copies is measured once, on that copy, rather than
    weighted by crystallographic multiplicity.
    """
    identifiers = entity["rcsb_polymer_entity_container_identifiers"]
    auth_chains = sorted(str(chain) for chain in identifiers["auth_asym_ids"])
    organisms = entity["rcsb_entity_source_organism"] or [{}]
    organism = organisms[0]
    return EntityMeta(
        entity_id=str(entity["rcsb_id"]),
        pdb_id=pdb_id,
        auth_chain=auth_chains[0],
        n_instances=len(auth_chains),
        sample_length_residues=int(entity["entity_poly"]["rcsb_sample_sequence_length"]),
        description=str(entity["rcsb_polymer_entity"]["pdbx_description"] or ""),
        organism=str(organism.get("ncbi_scientific_name") or ""),
        taxonomy_id=int(organism.get("ncbi_taxonomy_id") or -1),
        cluster_30=cluster_30(entity["rcsb_polymer_entity_group_membership"]),
    )


ENTRY_METADATA = "natural_binder_entries.json"


def fetch_entries(pdb_ids: list[str], progress: bool = True) -> dict[str, EntryMeta]:
    """Entry and entity metadata for every entry, in GraphQL batches."""
    recorded = paths.reference_table(ENTRY_METADATA)
    saved = json.loads(recorded.read_text()) if recorded.is_file() else {}
    entries = {
        key: EntryMeta(**{**value, "entities": tuple(EntityMeta(**e) for e in value["entities"])})
        for key, value in saved.items()
    }
    missing = sorted(set(pdb_ids) - entries.keys())
    for start in range(0, len(missing), GRAPHQL_BATCH):
        ids = missing[start : start + GRAPHQL_BATCH]
        payload = post_json(GRAPHQL_URL, {"query": GRAPHQL_TEMPLATE, "variables": {"ids": ids}})
        for entry in payload["data"]["entries"]:
            pdb_id = str(entry["rcsb_id"])
            info = entry["rcsb_entry_info"]
            resolutions = info["resolution_combined"] or [float("nan")]
            entries[pdb_id] = EntryMeta(
                pdb_id=pdb_id,
                title=str(entry["struct"]["title"] or ""),
                resolution_angstrom=float(resolutions[0]),
                deposit_date=str(entry["rcsb_accession_info"]["deposit_date"] or ""),
                release_date=str(entry["rcsb_accession_info"]["initial_release_date"] or ""),
                deposited_instances=int(info["deposited_polymer_entity_instance_count"]),
                entities=tuple(
                    parse_entity(pdb_id, entity) for entity in entry["polymer_entities"]
                ),
            )
        if progress:
            done = min(start + GRAPHQL_BATCH, len(pdb_ids))
            print(f"  metadata {done}/{len(pdb_ids)}", flush=True)
    if missing:
        recorded.parent.mkdir(parents=True, exist_ok=True)
        recorded.write_text(json.dumps({k: asdict(v) for k, v in entries.items()}, indent=2) + "\n")
    return {key: entries[key] for key in pdb_ids}


def download(pdb_id: str) -> Path:
    """Fetch one mmCIF into the cache, or return the cached copy.

    Files are kept gzipped as served. Measured on a representative entry the
    gzip is 3.7 times smaller than the text mmCIF and parses to identical
    composition, which turns a 1.5 GB download for the whole set into roughly
    400 MB.
    """
    CIF_CACHE.mkdir(parents=True, exist_ok=True)
    destination = CIF_CACHE / f"{pdb_id.upper()}.cif.gz"
    if destination.is_file() and destination.stat().st_size > 500:
        return destination
    url = FILE_URL.format(pdb_id=pdb_id.upper())
    last = ""
    for attempt in range(DOWNLOAD_ATTEMPTS):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                destination.write_bytes(response.read())
            return destination
        except Exception as error:  # noqa: BLE001
            last = str(error)
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"download failed for {pdb_id}: {last}")


def download_all(
    pdb_ids: list[str], progress: bool = True
) -> tuple[dict[str, Path], dict[str, str]]:
    """Fetch every mmCIF, returning the paths and any failures by entry."""
    found: dict[str, Path] = {}
    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        futures = {pool.submit(download, pdb_id): pdb_id for pdb_id in pdb_ids}
        for index, future in enumerate(futures, start=1):
            pdb_id = futures[future]
            try:
                found[pdb_id] = future.result()
            except Exception as error:  # noqa: BLE001
                failures[pdb_id] = str(error)
            if progress and index % 200 == 0:
                print(f"  downloaded {index}/{len(pdb_ids)}", flush=True)
    return found, failures


def _protein(cif_path: Path) -> struc.AtomArray:
    """Amino-acid atoms of model 1.

    Modified residues are kept. Dropping them, which an earlier version of this
    build did, splices non-adjacent residues together in the backbone array and
    so corrupts the i to i+4 relationships DSSP reads.
    """
    text = gzip.decompress(cif_path.read_bytes()).decode()
    array = pdbx.get_structure(pdbx.CIFFile.read(io.StringIO(text)), model=1)
    return array[struc.filter_amino_acids(array)]


def interface(protein: struc.AtomArray, binder_chain: str, target_chain: str) -> tuple[int, float]:
    """Heavy-atom contacts within the cutoff, and buried surface area."""
    binder = protein[protein.chain_id == binder_chain]
    target = protein[protein.chain_id == target_chain]
    distances = np.linalg.norm(binder.coord[:, None, :] - target.coord[None, :, :], axis=-1)
    contacts = int((distances < CONTACT_CUTOFF_ANGSTROM).sum())
    sasa_binder = np.nansum(struc.sasa(binder, point_number=SASA_POINT_NUMBER))
    sasa_target = np.nansum(struc.sasa(target, point_number=SASA_POINT_NUMBER))
    sasa_pair = np.nansum(struc.sasa(binder + target, point_number=SASA_POINT_NUMBER))
    return contacts, float(sasa_binder + sasa_target - sasa_pair)


def measure_entry(
    pdb_id: str, cif_path: Path, pairs: tuple[tuple[EntityMeta, EntityMeta], ...]
) -> EntryResult:
    """Measure composition and interface for every pair drawn from one entry."""
    result = EntryResult()
    try:
        protein = _protein(cif_path)
    except Exception as error:  # noqa: BLE001
        for binder, _ in pairs:
            result.failures[binder.entity_id] = f"parse: {error}"
        return result

    measured: dict[str, secondary_structure.Composition] = {}
    interfaces: dict[frozenset[str], tuple[int, float]] = {}
    for binder, target in pairs:
        try:
            for chain_id in (binder.auth_chain, target.auth_chain):
                if chain_id in measured:
                    continue
                selected = protein[protein.chain_id == chain_id]
                if selected.array_length() == 0:
                    raise ValueError(f"auth chain {chain_id} absent from model 1")
                measured[chain_id] = secondary_structure.composition(
                    secondary_structure.backbone_residues(selected)
                )
            key = frozenset({binder.auth_chain, target.auth_chain})
            if key not in interfaces:
                interfaces[key] = interface(protein, binder.auth_chain, target.auth_chain)
            contacts, bsa = interfaces[key]
        except Exception as error:  # noqa: BLE001
            result.failures[binder.entity_id] = str(error)
            continue
        binder_comp = measured[binder.auth_chain]
        target_comp = measured[target.auth_chain]
        result.rows.append(
            {
                "binder_entity_id": binder.entity_id,
                "binder_observed_length_residues": binder_comp.n_residues,
                "binder_n_helix_residues": binder_comp.n_helix,
                "binder_n_sheet_residues": binder_comp.n_sheet,
                "binder_n_coil_residues": binder_comp.n_coil,
                "binder_helix_fraction": binder_comp.helix_fraction,
                "binder_sheet_fraction": binder_comp.sheet_fraction,
                "binder_coil_fraction": binder_comp.coil_fraction,
                "binder_max_residue_gap_residues": binder_comp.max_residue_gap,
                "target_observed_length_residues": target_comp.n_residues,
                "target_helix_fraction": target_comp.helix_fraction,
                "target_sheet_fraction": target_comp.sheet_fraction,
                "interface_contacts_count": contacts,
                "interface_bsa_angstrom2": bsa,
            }
        )
    return result


def keyword_hit(text: str, keywords: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in keywords)


def pair_rows(
    hits: list[str], representatives: set[str], entries: dict[str, EntryMeta]
) -> pd.DataFrame:
    """One row per in-range entity, paired with the other entity of its entry."""
    hit_set = set(hits)
    rows: list[dict[str, object]] = []
    for entity_id in hits:
        pdb_id = entity_id.split("_")[0]
        entry = entries[pdb_id]
        binder = next(e for e in entry.entities if e.entity_id == entity_id)
        target = next(e for e in entry.entities if e.entity_id != entity_id)
        rows.append(
            {
                "pdb_id": pdb_id,
                "binder_entity_id": binder.entity_id,
                "binder_auth_chain": binder.auth_chain,
                "target_entity_id": target.entity_id,
                "target_auth_chain": target.auth_chain,
                "binder_sample_length_residues": binder.sample_length_residues,
                "target_sample_length_residues": target.sample_length_residues,
                "binder_instances_count": binder.n_instances,
                "target_instances_count": target.n_instances,
                "deposited_polymer_chains_count": entry.deposited_instances,
                "partner_in_range": int(target.entity_id in hit_set),
                "cluster_representative": int(entity_id in representatives),
                "binder_cluster_30": binder.cluster_30,
                "target_cluster_30": target.cluster_30,
                "resolution_angstrom": entry.resolution_angstrom,
                "deposit_date": entry.deposit_date,
                "release_date": entry.release_date,
                "binder_organism": binder.organism,
                "binder_taxonomy_id": binder.taxonomy_id,
                "target_organism": target.organism,
                "binder_description": binder.description,
                "target_description": target.description,
                "entry_title": entry.title,
                "binder_design_like": int(
                    keyword_hit(f"{entry.title} {binder.description}", DESIGN_KEYWORDS)
                ),
                "binder_engineered_scaffold_like": int(
                    keyword_hit(binder.description, ENGINEERED_SCAFFOLD_KEYWORDS)
                ),
                "target_engineered_scaffold_like": int(
                    keyword_hit(target.description, ENGINEERED_SCAFFOLD_KEYWORDS)
                ),
            }
        )
    return pd.DataFrame(rows)


def search_hits(progress: bool = True) -> tuple[list[str], set[str], int]:
    """The in-range entities, the cluster representatives, and the cluster count.

    Read from the recorded result sets if they are on disk, otherwise fetched
    from the live API and recorded. Reading them is what makes a rebuild
    reproducible; fetching them is what makes the set derivable in the first
    place.
    """
    recorded = paths.reference_table(SEARCH_HITS)
    if recorded.is_file():
        payload = json.loads(recorded.read_text())
        if progress:
            print(
                f"using the recorded search results in {recorded.name}, "
                f"snapshot {payload['snapshot_release_date']}"
            )
        return (
            list(payload["hits"]),
            set(payload["representatives"]),
            int(payload["clusters"]),
        )

    if progress:
        print(f"querying RCSB, released on or before {SNAPSHOT_RELEASE_DATE}...")
    all_hits = post_json(SEARCH_URL, search_body(False))
    hits = [hit["identifier"] for hit in all_hits["result_set"]]
    grouped = post_json(SEARCH_URL, search_body(True))
    representatives = {hit["identifier"] for hit in grouped["result_set"]}
    n_clusters = int(grouped["group_by_count"])
    if progress:
        print(
            f"  {len(hits)} in-range entities, {len(representatives)} cluster "
            f"representatives of {n_clusters} clusters"
        )
    recorded.parent.mkdir(parents=True, exist_ok=True)
    (paths.REFERENCE / "natural_binder_query.json").write_text(
        json.dumps(search_body(True), indent=2) + "\n"
    )
    (paths.REFERENCE / "natural_binder_query_all_hits.json").write_text(
        json.dumps(search_body(False), indent=2) + "\n"
    )

    recorded.write_text(
        json.dumps(
            {
                "snapshot_release_date": SNAPSHOT_RELEASE_DATE,
                "clusters": n_clusters,
                "representatives": sorted(representatives),
                "hits": hits,
            },
            indent=2,
        )
        + "\n"
    )
    return hits, representatives, n_clusters


def build(
    limit: int = 0,
    progress: bool = True,
    *,
    include_excluded: bool = False,
    workers: int = DEFAULT_WORKERS,
) -> pd.DataFrame:
    """Recalculate reference measurements, returning results without caching them.

    ``limit`` selects a small subset for development. ``include_excluded``
    returns the full pair table, with selection flags, instead of retained pairs.
    Frozen search and entry metadata and downloaded CIFs are preserved as inputs.
    Measurements use spawned processes, including from notebooks; ``workers=1``
    runs serially. Downloads use their separate thread pool before measurements.
    """
    validate_workers(workers)
    paths.REFERENCE.mkdir(parents=True, exist_ok=True)
    hits, representatives, n_clusters = search_hits(progress=progress)

    if limit:
        representatives = set(sorted(representatives)[:limit])
        keep = {entity.split("_")[0] for entity in representatives}
        hits = [hit for hit in hits if hit.split("_")[0] in keep]
        if progress:
            print(f"  limited to {len(representatives)} representatives")

    pdb_ids = sorted({hit.split("_")[0] for hit in hits})
    entries = fetch_entries(pdb_ids, progress=progress)
    pairs = pair_rows(hits, representatives, entries)

    to_measure = pairs[pairs["cluster_representative"] == 1]
    if progress:
        print(f"downloading {to_measure['pdb_id'].nunique()} mmCIF files...")
    found, download_failures = download_all(sorted(set(to_measure["pdb_id"])), progress=progress)

    if progress:
        print(f"measuring {len(found)} entries...")
    rows: list[dict[str, object]] = []
    failures: dict[str, str] = {
        f"{pdb_id}_download": error for pdb_id, error in download_failures.items()
    }
    jobs = []
    for pdb_id, group in to_measure.groupby("pdb_id", sort=True):
        if pdb_id not in found:
            continue
        by_id = {entity.entity_id: entity for entity in entries[pdb_id].entities}
        entry_pairs = tuple(
            (by_id[str(record["binder_entity_id"])], by_id[str(record["target_entity_id"])])
            for record in group.to_dict("records")
        )
        jobs.append((str(pdb_id), found[pdb_id], entry_pairs))
    for index, result in enumerate(starmap(measure_entry, jobs, workers=workers), start=1):
        rows.extend(result.rows)
        failures.update(result.failures)
        if progress and index % 200 == 0:
            print(f"  measured {index} entries ({len(failures)} failures)", flush=True)

    pairs = pairs.merge(pd.DataFrame(rows), on="binder_entity_id", how="left")
    pairs["measured"] = pairs["binder_observed_length_residues"].notna().astype(int)
    pairs["binder_observed_fraction"] = (
        pairs["binder_observed_length_residues"] / pairs["binder_sample_length_residues"]
    )
    pairs["excluded_short_or_long"] = (
        (pairs["binder_observed_length_residues"] < MIN_LENGTH_RESIDUES)
        | (pairs["binder_observed_length_residues"] > MAX_LENGTH_RESIDUES)
    ).astype(int)
    pairs["excluded_incomplete"] = (
        pairs["binder_observed_fraction"] < MIN_OBSERVED_FRACTION
    ).astype(int)
    pairs["flag_small_interface"] = (
        pairs["interface_bsa_angstrom2"] < MIN_PLAUSIBLE_BSA_ANGSTROM2
    ).astype(int)
    pairs["flag_peptide_target"] = (
        pairs["target_sample_length_residues"] < MIN_TARGET_LENGTH_RESIDUES
    ).astype(int)
    pairs["in_reference_set"] = (
        (pairs["cluster_representative"] == 1)
        & (pairs["measured"] == 1)
        & (pairs["binder_design_like"] == 0)
        & (pairs["excluded_short_or_long"] == 0)
        & (pairs["excluded_incomplete"] == 0)
    ).astype(int)
    pairs = pairs.sort_values("binder_entity_id").reset_index(drop=True)

    table = pairs[pairs["in_reference_set"] == 1].reset_index(drop=True)

    counts = {
        "snapshot_release_date": SNAPSHOT_RELEASE_DATE,
        "in_range_entities": len(hits),
        "in_range_entries": len(pdb_ids),
        "clusters": n_clusters,
        "cluster_representatives": int((pairs["cluster_representative"] == 1).sum()),
        "measured": int((pairs["measured"] == 1).sum()),
        "excluded_design_like": int(
            ((pairs["cluster_representative"] == 1) & (pairs["binder_design_like"] == 1)).sum()
        ),
        "excluded_short_or_long": int(
            ((pairs["measured"] == 1) & (pairs["excluded_short_or_long"] == 1)).sum()
        ),
        "excluded_incomplete": int(
            ((pairs["measured"] == 1) & (pairs["excluded_incomplete"] == 1)).sum()
        ),
        "flagged_small_interface": int(
            ((pairs["in_reference_set"] == 1) & (pairs["flag_small_interface"] == 1)).sum()
        ),
        "flagged_peptide_target": int(
            ((pairs["in_reference_set"] == 1) & (pairs["flag_peptide_target"] == 1)).sum()
        ),
        "final_reference_pairs": int(len(table)),
        "measurement_failures": len(failures),
    }
    if progress:
        print(json.dumps(counts, indent=2))
    return pairs if include_excluded else table
