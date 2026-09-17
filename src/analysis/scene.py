"""Interactive molecular scene for the secondary-structure loss term.

The term this publication steers with is a plain mean: every residue's i to
i + 4 contact probability is averaged, and that average is the penalty. The
point the scene has to carry is that there is no selection step, so a reader
can hover any residue in the binder and find it contributing on the same
footing as every other.

Contact probability comes from the distogram during optimization and is not
recoverable from a deposited structure, so the scene reads the geometry
instead: a rung counts when its Ca atoms sit closer than the same cutoff the
term uses. That makes the displayed value the term evaluated with hard zero-one
probabilities, which is the quantity the term's own docstring calls the
expected helical-rung fraction.

The structure the scene loads is the designed binder on its own. The complex it
came from also held the target the binder was designed against, which owns no
i to i + 4 rung the term ever sees and which, drawn even at low opacity, stood
between the cursor and the helices the scene exists to show. It is gone from the
asset rather than styled faintly.
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

CONTRIBUTING_COLOR = apc.tanzanite.hex_code
ZERO_COLOR = apc.ghost.hex_code
"""What the scene encodes: what each residue contributes to the term.

Not secondary structure. The publication's three state hues are deliberately
absent here, because a residue's contribution is what the term reads and a
state is not: colouring by state would present a classification the term never
makes, and would invite the helix hue to be read as the reason a residue
scores.

A residue contributes either one over the number of scored residues or exactly
zero, with nothing in between, so these are a pair from one hue family
separated by lightness rather than a ramp. The trailing residues have no i+4
partner and take the same colour as a scored residue that contributes nothing,
because the panel is about what reaches the sum and by that measure they are
alike.

The two are separated by 55.2 under the worst of normal vision, protanopia and
deuteranopia, and by at least 18.6 from every hue already carrying meaning
elsewhere in the publication.

The pale one is a cool near-white, 2.0 from the background the viewer draws on.
That is deliberate and is the one place in this publication where a colour is
not readable against its surface on its own: the cartoon is lit and shadowed,
so the ribbon reads as geometry rather than as fill, and a residue contributing
nothing should recede to the point of being scenery. Anything encoded only by
this colour would be invisible, so nothing is.
"""
ACCENT_COLOR = apc.charcoal.hex_code
SURFACE_COLOR = "#FEF7F1"

ENDPOINT_COLOR = "#F7B846"
"""Canary, the same at both ends of a rung whether or not the rung counts.

Taking the endpoint colour from the cartoon hid the endpoints, because a sphere
sits on the very residue whose colour it was borrowing. Canary is the house
colour that stays furthest from everything a sphere has to be read against
here. Its smallest CAM02-UCS difference is 37.9 from the contact red,
34.9 from the non-contact blue-grey, 53.1 from the charcoal rod it caps and 29.4
from the white behind the binder's silhouette, each of those the worst case over
normal vision and fully simulated deuteranopia, protanopia and tritanopia. It
shares a hue family with the mustard the composition panels give to sheet, which
costs nothing here: no sheet colour appears in this scene, and the spheres mark
a measurement rather than a structural state.
"""

BINDER_CHAIN = "B"
ENDPOINT_RADIUS = 1.65
CYLINDER_RADIUS = 0.3
DASH_LENGTH = 0.55
GAP_LENGTH = 0.4
"""Dash geometry for the rod between the two endpoints.

3Dmol defaults both to 0.25, which at this rod thickness renders as a line of
beads rather than a dashed rod. Lengthening the dash past the gap keeps it
reading as a broken line at a radius chosen to sit against 1.65 A spheres.
"""
LABEL_FONT_SIZE = 13
LABEL_PAD = 12
VIEW_WIDTH = "100%"
"""Viewer width as a CSS length, so the scene tracks the article body.

py3Dmol writes an integer straight into the container style as pixels, which
leaves a viewer wider than the body overflowing it, and a molecule centred in
the canvas then reads as off-centre because part of the canvas is not on
screen. A percentage makes the container track whatever column it is placed in.
"""

VIEW_HEIGHT = 560
"""Viewer height in pixels, which stays fixed because nothing constrains it.

The body sets the width; height has no equivalent to track, and a percentage
height inside an auto-height block collapses to nothing.
"""

LABEL_MARGIN = 10
"""Inset of the pinned measurement from the top left corner, in screen pixels."""

LABEL_BOX_PADDING = 4
"""Pixels between a row's text and the edge of the row's own background box.

This is 3Dmol's own default, restated here because the row spacing below is
derived from it, and a default that moved underneath us would open gaps between
the rows.
"""

LABEL_LINE_HEIGHT = LABEL_FONT_SIZE * 1.25 + 2 * LABEL_BOX_PADDING
"""How far apart to stack the rows of the hover label, in screen pixels.

3Dmol measures a row of text, sizes a canvas as ``fontSize * 1.25 + 2 *
padding`` with the 1.25 leaving room for descenders, and draws that canvas at
exactly its own pixel size. Offsetting each row by that height therefore butts
the rows against each other with neither a gap nor an overlap, so a stack of
one-line labels reads as one block of text.
"""

HOVER_DURATION_MS = 0
"""Show the measurement on the first frame the cursor is over a residue.

py3Dmol defaults to 500 ms, which reads as the scene ignoring the cursor. The
handler only adds three shapes and a short stack of one-line labels, so there is
no cost to hiding behind a delay.
"""


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
    radius: __CYLINDER_RADIUS__, color: '__ACCENT__', dashed: true,
    dashLength: __DASH_LENGTH__, gapLength: __GAP_LENGTH__, fromCap: 1, toCap: 1});
  atom.__s1 = viewer.addSphere({center: {x: r.a[0], y: r.a[1], z: r.a[2]},
    radius: __ENDPOINT_RADIUS__, color: '__ENDPOINT_COLOR__'});
  atom.__s2 = viewer.addSphere({center: {x: r.b[0], y: r.b[1], z: r.b[2]},
    radius: __ENDPOINT_RADIUS__, color: '__ENDPOINT_COLOR__'});
  stack([
    pad('i') + '= ' + atom.resi,
    pad('i + 4') + '= ' + r.j,
    pad('Distance') + '= ' + r.d.toFixed(2) + ' ' + ANGSTROM,
    pad('Counts') + '= ' + (near ? 'yes  (< __CUTOFF__ ' + ANGSTROM + ')'
                                 : 'no  (>= __CUTOFF__ ' + ANGSTROM + ')'),
    pad('Adds') + '= ' + (near ? r.c.toFixed(4) : '0.0000')
  ]);
  viewer.render();
}"""
"""The hover handler, with placeholders substituted rather than interpolated.

The body is JavaScript and is full of braces, so both percent formatting and
str.format need every one of them escaped or doubled. Named placeholders keep
the template readable as the code it actually is.

Every row of the measurement is its own label. 3Dmol sizes a label from a single
``measureText`` and paints it with a single ``fillText``, so a newline in the
text is not a break but one more unrenderable character in a one-line box, and
the rows arrive side by side however the newline is spelled. Stacking separate
labels makes the breaks structural instead. The handles go on the atom in an
array because the unhover callback is handed the atom and nothing else, and has
to be able to take back every label this one added.

The stack is pinned to the top left of the viewport with ``useScreen`` rather
than anchored to the rung it describes. The pixel coordinates go in ``position``
even in this mode, because ``useScreen`` only tells the sprite material to read
that same field as an offset from the upper left instead of as a point in the
scene; a spec carrying bare ``x`` and ``y`` keys silently falls back to the
default position of ``{x: -10, y: 1, z: 1}``, which stacks every row on one spot
off the left edge of the view. Anchored to the structure it sat on top
of the two spheres it was labelling, most often the one at residue i, so reading
the measurement hid the thing being measured. A fixed corner also removes the
one clipping case an anchored stack had, where a rung near the top of the view
pushed its own rows off the edge.

Each row carries its own background box, sized to its own text, so the rows are
padded out to the longest of them before they are drawn. The font is monospace,
so that leaves every box the same width and the stack a straight edge instead of
a ragged one.

The angstrom sign is built with ``String.fromCharCode`` rather than written as
an escape, because py3Dmol escapes the callback again when it embeds it in the
page and a backslash escape arrives at the browser as its own literal
characters.
"""


def _binder(path: Path) -> struc.AtomArray:
    """The designed chain, amino acids only."""
    array = pdb.PDBFile.read(str(path)).get_structure(model=1)
    array = array[struc.filter_amino_acids(array)]
    return array[array.chain_id == BINDER_CHAIN]


def _rungs(path: Path) -> tuple[dict[int, dict[str, object]], int]:
    """Every i to i + 4 Ca pair in the binder, with the geometry to draw it."""
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
    """A hoverable binder, every residue showing what it adds to the term.

    The view is centred and zoomed on the binder, and ``center`` is called as
    well as ``zoomTo`` so that the point the scene rotates about is the middle
    of the chain rather than wherever the model's coordinates happen to sit.
    ``resize`` follows both, because the container takes its width from the
    page and 3Dmol has to measure the canvas after the layout settles.

    The binder is coloured by what each residue contributes to the term, so the
    stretches the term charges for are visible before the cursor goes anywhere.
    Hovering draws the pair being measured as two spheres and a dashed rod, and
    writes the distance, whether the rung counts, and the share it contributes.
    """
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
    view.resize()
    return view
