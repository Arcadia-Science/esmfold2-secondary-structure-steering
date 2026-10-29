"""Spawned structural measurements must match serial and notebook execution."""

from __future__ import annotations
import gzip
import io
import json
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest
from analysis import paths, rcsb, secondary_structure
from analysis.parallel import starmap
from biotite.structure.io import pdb, pdbx


@pytest.fixture
def reference_inputs(tmp_path, monkeypatch):
    # A tracked binder with both helix and sheet, copied into two chains.
    # Absolute paths and metadata are passed to workers; no inherited monkeypatches.
    protein = pdb.get_structure(pdb.PDBFile.read(paths.ASSETS / "example-complex.pdb"), model=1)
    partner = protein.copy()
    partner.chain_id[:] = "A"
    partner.coord += np.array([8.0, 0.0, 0.0])
    cif = pdbx.CIFFile()
    pdbx.set_structure(cif, protein + partner)
    buffer = io.StringIO()
    cif.write(buffer)
    cache = tmp_path / "cifs"
    cache.mkdir()
    good = cache / "GOOD.cif.gz"
    good.write_bytes(gzip.compress(buffer.getvalue().encode()))
    bad = cache / "BAD.cif.gz"
    bad.write_bytes(b"invalid gzip")
    length = len(secondary_structure.backbone_residues(protein))
    entities = tuple(
        rcsb.EntityMeta(f"{name}_1", name, "B", 1, length, "natural", "species", 1, "c1")
        for name in ("GOOD", "BAD")
    )
    entries = {
        name: rcsb.EntryMeta(
            name,
            "natural complex",
            2.0,
            "2020-01-01",
            "2020-01-01",
            2,
            (
                entity,
                rcsb.EntityMeta(f"{name}_2", name, "A", 1, length, "partner", "species", 1, "c2"),
            ),
        )
        for name, entity in zip(("GOOD", "BAD"), entities, strict=True)
    }
    (tmp_path / rcsb.ENTRY_METADATA).write_text(
        json.dumps({key: asdict(value) for key, value in entries.items()})
    )
    (tmp_path / rcsb.SEARCH_HITS).write_text(
        json.dumps(
            {
                "hits": [e.entity_id for e in entities],
                "representatives": [e.entity_id for e in entities],
                "clusters": 2,
                "snapshot_release_date": rcsb.SNAPSHOT_RELEASE_DATE,
            }
        )
    )
    monkeypatch.setattr(paths, "REFERENCE", tmp_path)
    monkeypatch.setattr(rcsb, "CIF_CACHE", cache)
    return tmp_path


def test_parallel_reference_preserves_rows_flags_and_failures(reference_inputs):
    serial = rcsb.build(progress=False, include_excluded=True, workers=1)
    parallel = rcsb.build(progress=False, include_excluded=True, workers=2)
    pd.testing.assert_frame_equal(serial, parallel, check_exact=True)
    assert serial.set_index("pdb_id").measured.to_dict() == {"BAD": 0, "GOOD": 1}
    assert serial.set_index("pdb_id").in_reference_set.to_dict() == {"BAD": 0, "GOOD": 1}


def test_empty_tasks_do_not_start_a_pool():
    assert list(starmap(rcsb.measure_entry, [], workers=2)) == []


def test_spawned_measurements_execute_in_a_notebook(reference_inputs):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    source = paths.REPO_ROOT / "src"
    code = f"""
import multiprocessing
import sys
from pathlib import Path
sys.path.insert(0, {str(source)!r})
import pandas as pd
from analysis import paths, rcsb, reference
paths.REFERENCE = Path({str(reference_inputs)!r})
rcsb.CIF_CACHE = paths.REFERENCE / "cifs"
serial = reference.composition(workers=1)
parallel = reference.composition(workers=2)
pd.testing.assert_frame_equal(serial, parallel, check_exact=True)
assert not multiprocessing.active_children(), "Measurement pools leaked workers"
print("Notebook spawn and exact equality verified")
"""
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(code)])
    executed = nbclient.NotebookClient(
        notebook,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(reference_inputs)}},
    ).execute()
    assert any(
        "Notebook spawn and exact equality verified" in output.get("text", "")
        for output in executed.cells[0].outputs
    )
