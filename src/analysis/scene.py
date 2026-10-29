"""Illustrate binder i to i+4 contacts from deposited coordinates.

The display uses hard distance thresholds as a geometric illustration of the
soft distogram loss; it does not recover optimization-time probabilities.
"""

from __future__ import annotations
import json
from pathlib import Path

import arcadia_pycolor as apc
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import numpy as np
import py3Dmol

from analysis import steering

CONTRIBUTING_COLOR = apc.cinnabar.hex_code
ZERO_COLOR = apc.ghost.hex_code
"""Colours for counted and uncounted geometric contacts, including unscored termini."""
ACCENT_COLOR = apc.charcoal.hex_code
SURFACE_COLOR = "#FEF7F1"

ENDPOINT_COLOR = apc.sky.hex_code
"""Colour of contact endpoint spheres and the rod connecting them."""

BINDER_CHAIN = "B"
ENDPOINT_RADIUS = 1.65
CYLINDER_RADIUS = 0.3
DASH_LENGTH = 0.55
GAP_LENGTH = 0.4
"""Gap length between contact-rod dashes."""
LABEL_FONT_SIZE = 13
LABEL_PAD = 12
VIEW_WIDTH = "100%"
"""Viewer width as a CSS length."""

VIEW_HEIGHT = 560
"""Viewer height in pixels."""

ZOOM_DISTANCE_LIMITS = (25, 250)
"""Camera-distance bounds in angstroms."""

LABEL_MARGIN = 10
"""Inset of the pinned measurement from the top left corner, in screen pixels."""

LABEL_BOX_PADDING = 4
"""Padding around each hover-label row, in pixels."""

LABEL_LINE_HEIGHT = LABEL_FONT_SIZE * 1.25 + 2 * LABEL_BOX_PADDING
"""Vertical spacing between hover-label rows, in pixels."""

HOVER_DURATION_MS = 0
"""Delay before displaying a hover measurement, in milliseconds."""


HOVER_TEMPLATE = """function(atom, viewer) {
  var ANGSTROM = String.fromCharCode(197);
  if (typeof window.__RUNGS === 'undefined') { window.__RUNGS = __RUNGS__; }
  if (atom.__labs) { return; }
  var pad = function (k) { while (k.length < __PAD__) { k += ' '; } return k; };
  var stack = function (rows) {
    var width = rows.reduce(function (n, row) { return Math.max(n, row.length); }, 0);
    atom.__labs = rows.map(function (row, index) {
      while (row.length < width) { row += ' '; }
      return viewer.addLabel(row, {
        useScreen: true,
        position: {x: __MARGIN__, y: __MARGIN__ + index * __LINE_HEIGHT__, z: 0},
        backgroundColor: '__SURFACE__', backgroundOpacity: 0.95,
        fontColor: '__ACCENT__', font: 'monospace', fontSize: __FONT_SIZE__,
        padding: __BOX_PADDING__, borderThickness: 0, alignment: 'topLeft'});
    });
  };
  var r = window.__RUNGS[atom.resi];
  if (!r) {
    stack([pad('i') + '= ' + atom.resi, pad('i + 4') + '= none']);
    viewer.render();
    return;
  }
  var near = r.d < __CUTOFF__;
  var mid = {x: (r.a[0] + r.b[0]) / 2, y: (r.a[1] + r.b[1]) / 2, z: (r.a[2] + r.b[2]) / 2};
  atom.__cyl = viewer.addCylinder({
    start: {x: r.a[0], y: r.a[1], z: r.a[2]}, end: {x: r.b[0], y: r.b[1], z: r.b[2]},
    radius: __CYLINDER_RADIUS__, color: '__ENDPOINT_COLOR__', dashed: true,
    dashLength: __DASH_LENGTH__, gapLength: __GAP_LENGTH__, fromCap: 1, toCap: 1});
  atom.__s1 = viewer.addSphere({center: {x: r.a[0], y: r.a[1], z: r.a[2]},
    radius: __ENDPOINT_RADIUS__, color: '__ENDPOINT_COLOR__'});
  atom.__s2 = viewer.addSphere({center: {x: r.b[0], y: r.b[1], z: r.b[2]},
    radius: __ENDPOINT_RADIUS__, color: '__ENDPOINT_COLOR__'});
  stack([
    pad('i') + '= ' + atom.resi,
    pad('i + 4') + '= ' + r.j,
    pad('Distance') + '= ' + r.d.toFixed(2) + ' ' + ANGSTROM,
    pad('Contributes') + '= ' + (near ? 'yes  (< __CUTOFF__ ' + ANGSTROM + ')'
                                 : 'no  (>= __CUTOFF__ ' + ANGSTROM + ')'),
    pad('Loss') + '= ' + (near ? r.c.toFixed(4) : '0.0000')
  ]);
  viewer.render();
}"""
"""Hover callback template with named placeholders.

Stack monospace labels at the top left and retain their handles for cleanup.
"""


def _binder(path: Path) -> struc.AtomArray:
    """The designed chain, amino acids only."""
    array = pdb.PDBFile.read(str(path)).get_structure(model=1)
    array = array[struc.filter_amino_acids(array)]
    return array[array.chain_id == BINDER_CHAIN]


def _rungs(path: Path) -> tuple[dict[int, dict[str, object]], int]:
    """Pair binder Cα atoms four positions apart in coordinate order for display."""
    binder = _binder(path)
    alpha = binder[binder.atom_name == "CA"]
    coords, residues = alpha.coord, alpha.res_id

    offset = steering.HELIX_PENALTY_OFFSET
    distance = np.linalg.norm(coords[offset:] - coords[:-offset], axis=1)
    scored = len(distance)
    return {
        int(residues[index]): {
            "d": round(float(distance[index]), 2),
            "j": int(residues[index + offset]),
            "a": [round(float(value), 3) for value in coords[index]],
            "b": [round(float(value), 3) for value in coords[index + offset]],
            "c": round(1.0 / scored, 4),
        }
        for index in range(scored)
    }, scored


def loss_term_scene(path: Path) -> py3Dmol.view:
    """Show binder contact contributions with hoverable distances and endpoint markers."""
    rungs, scored = _rungs(path)
    cutoff = steering.HELIX_PENALTY_CUTOFF_ANGSTROM

    hover = (
        HOVER_TEMPLATE.replace("__RUNGS__", json.dumps(rungs))
        .replace("__PAD__", str(LABEL_PAD))
        .replace("__SURFACE__", SURFACE_COLOR)
        .replace("__ACCENT__", ACCENT_COLOR)
        .replace("__FONT_SIZE__", str(LABEL_FONT_SIZE))
        .replace("__BOX_PADDING__", str(LABEL_BOX_PADDING))
        .replace("__MARGIN__", str(LABEL_MARGIN))
        .replace("__LINE_HEIGHT__", str(LABEL_LINE_HEIGHT))
        .replace("__CUTOFF__", str(cutoff))
        .replace("__CYLINDER_RADIUS__", str(CYLINDER_RADIUS))
        .replace("__DASH_LENGTH__", str(DASH_LENGTH))
        .replace("__GAP_LENGTH__", str(GAP_LENGTH))
        .replace("__ENDPOINT_RADIUS__", str(ENDPOINT_RADIUS))
        .replace("__ENDPOINT_COLOR__", ENDPOINT_COLOR)
    )

    unhover = """function(atom, viewer) {
  ['__cyl', '__s1', '__s2'].forEach(function (k) {
    if (atom[k]) { viewer.removeShape(atom[k]); delete atom[k]; }
  });
  if (atom.__labs) {
    atom.__labs.forEach(function (label) { viewer.removeLabel(label); });
    delete atom.__labs;
  }
  viewer.render();
}"""

    view = py3Dmol.view(width=VIEW_WIDTH, height=VIEW_HEIGHT)
    view.addModel(path.read_text(), "pdb")
    contributing = [key for key, rung in rungs.items() if float(rung["d"]) < cutoff]
    view.setStyle({"chain": BINDER_CHAIN}, {"cartoon": {"color": ZERO_COLOR}})
    if contributing:
        view.addStyle(
            {"chain": BINDER_CHAIN, "resi": contributing},
            {"cartoon": {"color": CONTRIBUTING_COLOR}},
        )
    view.setHoverable({"chain": BINDER_CHAIN}, True, hover, unhover)
    view.setHoverDuration(HOVER_DURATION_MS)
    view.center({"chain": BINDER_CHAIN})
    view.zoomTo({"chain": BINDER_CHAIN})
    view.setZoomLimits(*ZOOM_DISTANCE_LIMITS)
    view.resize()
    return view
