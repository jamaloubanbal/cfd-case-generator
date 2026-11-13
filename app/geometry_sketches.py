"""
ASCII-art geometry sketches for the Streamlit sidebar.

Each entry in GEOMETRY_SKETCHES maps geometry_type →
    (sketch: str, description: str)

Sketches use Unicode box-drawing characters and are designed
to render correctly inside st.code() (monospace, ~26 chars wide).
"""

# ---------------------------------------------------------------------------
# Sketch definitions — one per geometry type
# ---------------------------------------------------------------------------
#
# Layout conventions:
#   →  ←  ↑  ↓  ↗  ↘  ↙  ↖   flow-direction arrows
#   ┌ ┐ └ ┘ │ ─   box-drawing (thin)
#   ┤ ├ ┬ ┴ ┼     T/cross junctions
#   ╔ ╗ ╚ ╝ ║ ═   box-drawing (double, atmosphere boundary)
#   ▓            filled water region (dambreak)
#   /            obstacle fill

GEOMETRY_SKETCHES: dict[str, tuple[str, str]] = {

    # ------------------------------------------------------------------
    # channel — simple rectangular duct, inlet left, outlet right
    # ------------------------------------------------------------------
    "channel": (
        """\
       walls (top)
 ┌─────────────────────┐
→│  →    →    →    →  →│→
 │                     │
→│  →    →    →    →  →│→
 └─────────────────────┘
       walls (bot)
 inlet               outlet""",
        "Rectangular channel — inlet left, outlet right",
    ),

    # ------------------------------------------------------------------
    # cavity — square box, hot wall left, cold wall right.
    #   Flow: hot left → air rises on left, crosses top rightward,
    #         descends near cold right, returns along bottom leftward.
    #   Single counterclockwise vortex (viewed x→, y↑).
    # ------------------------------------------------------------------
    "cavity": (
        """\
       top (wall)
 ┌──────────────────┐
 │ ↗  →   →   →  ↘ │
H│↑                ↓│C
 │↑                ↓│
H│↑                ↓│C
 │ ↖  ←   ←   ←  ↙ │
 └──────────────────┘
     bottom (wall)
hotWall          coldWall""",
        "Square cavity — natural convection, hot/cold side walls",
    ),

    # ------------------------------------------------------------------
    # t_junction — inlet1 (left) + inlet2 (branch top) → outlet (right)
    # Block layout: branch above centre; main channel B-left | B-mid | B-right
    # ------------------------------------------------------------------
    "t_junction": (
        """\
          inlet2
            ↓
      ┌─────┐
      │  ↓  │
┌─────┘  ↓  └──────────┐
│  →  →  ↓  →  →  →  →│→
└──────────────────────┘
inlet1   walls    outlet""",
        "T-junction — two inlets merging to one outlet",
    ),

    # ------------------------------------------------------------------
    # rectangular_obstacle — channel with central blockage (8-block mesh)
    #   top/bottom walls; obstacle patch on all 4 faces of the block
    # ------------------------------------------------------------------
    "rectangular_obstacle": (
        """\
      walls (top)
 ┌──────────────────────┐
→│  →  →  ┌─────┐  →  →│→
 │        │/////│       │
→│  →  →  └─────┘  →  →│→
 └──────────────────────┘
      walls (bot)
 inlet  obstacle  outlet""",
        "Channel with central rectangular obstacle",
    ),

    # ------------------------------------------------------------------
    # backstep — backward-facing step (sudden expansion downward).
    #   Inlet channel occupies y:[Hs,H]; after step, full height y:[0,H].
    #   Top wall is continuous; bottom wall starts at the step face.
    # ------------------------------------------------------------------
    "backstep": (
        """\
 ┌──────────┬─────────────────┐
→│  →  →  →│   →   →   →  → │→
 └──────────┘                 │
   inlet    step     outlet   │
             └────────────────┘
              walls (floor)""",
        "Backward-facing step — sudden downward expansion",
    ),

    # ------------------------------------------------------------------
    # dambreak — rectangular basin, atmosphere on top, initial water
    #            column on left side; interFoam two-phase VOF
    # ------------------------------------------------------------------
    "dambreak": (
        """\
         atmosphere
 ╔═══════════════════╗
→║ ▓▓▓▓▓│            ║
 ║ ▓▓▓▓▓│   air      ║
 ║ ▓▓▓▓▓│            ║
 ╚═══════════════════╝
       walls (floor)
 inlet  water│air   outlet""",
        "Dam-break basin — two-phase VOF free surface",
    ),

    # ------------------------------------------------------------------
    # cylinder — 2-D cylinder in cross-flow, O-grid / butterfly mesh.
    #   top/bottom are symmetry planes; cylinder is a wall patch.
    #   Flow goes around the cylinder; wake forms downstream.
    # ------------------------------------------------------------------
    "cylinder": (
        """\
      symmetry (top)
 ┌─────────────────────┐
→│  →  →  .--.  →  →  │→
→│  →  → ( ○  ) →  →  │→
→│  →  →  `--'  →  →  │→
 └─────────────────────┘
      symmetry (bot)
 inlet  cylinder  outlet""",
        "2-D cylinder in cross-flow — O-grid butterfly mesh",
    ),

    # ------------------------------------------------------------------
    # pipe — axisymmetric pipe flow, 5° wedge mesh.
    #   Axis runs along x; y is the radial direction (0 → R).
    #   Only a 5° wedge sector is meshed; front/back are wedge patches.
    #   The axis patch (r=0) is the degenerate lower edge of the wedge.
    # ------------------------------------------------------------------
    "pipe": (
        """\
         wall (r = R)
 ┌───────────────────────┐
→│  →    →    →    →   →│→
 ├╌╌╌╌╌╌ axis (r=0) ╌╌╌╌┤
→│  →    →    →    →   →│→
 └───────────────────────┘
         wall (r = R)
 inlet  [wedge ±2.5°]  outlet""",
        "Axisymmetric pipe flow — 5° wedge mesh",
    ),

    # ------------------------------------------------------------------
    # airfoil — 2-D airfoil in external flow, 8-block C/H-grid.
    #   top/bottom are symmetry planes; airfoil body is a rectangular
    #   hole in the mesh (no cells inside); wall BCs on LE, upper,
    #   lower, and TE faces.  Far-field inlet on left, outlet on right.
    # ------------------------------------------------------------------
    "airfoil": (
        """\
       top (symmetry)
 ┌──────────────────────────┐
→│  →  →  ╔══════╗  →   →  │→
 │         ║      ║  wake   │
→│  →  →  ╚══════╝  →   →  │→
 └──────────────────────────┘
       bottom (symmetry)
 inlet  airfoil  TE   outlet""",
        "2-D airfoil — 8-block C/H-grid, rectangular body hole",
    ),

    # ------------------------------------------------------------------
    # stl_import — arbitrary STL body inside a background box mesh;
    #              snappyHexMesh refines the background mesh around it
    # ------------------------------------------------------------------
    "stl_import": (
        """\
 ┌──────background──────┐
 │                      │
→│  ┌───refinement───┐  │→
 │  │  ╔══body══╗   │  │
→│  └────────────────┘  │→
 │                      │
 └──────symmetry────────┘
inlet      STL      outlet""",
        "Custom STL geometry — snappyHexMesh background + surface refinement",
    ),
}

# Human-readable labels for each geometry type
GEOMETRY_LABELS: dict[str, str] = {
    "channel":              "Rectangular Channel",
    "cavity":               "Square Cavity",
    "t_junction":           "T-Junction",
    "rectangular_obstacle": "Channel + Obstacle",
    "backstep":             "Backward-Facing Step",
    "dambreak":             "Dam-Break Basin",
    "cylinder":             "2-D Cylinder (O-grid)",
    "pipe":                 "Axisymmetric Pipe (Wedge)",
    "airfoil":              "2-D Airfoil (C/H-grid)",
    "stl_import":           "Custom STL (snappyHexMesh)",
}


def get_sketch(geometry_type: str) -> tuple[str, str]:
    """Return ``(sketch_str, description)`` for *geometry_type*.

    Falls back to ``"channel"`` if *geometry_type* is unknown.
    """
    return GEOMETRY_SKETCHES.get(geometry_type, GEOMETRY_SKETCHES["channel"])


def get_label(geometry_type: str) -> str:
    """Return a human-readable label for *geometry_type*."""
    return GEOMETRY_LABELS.get(geometry_type, geometry_type.replace("_", " ").title())
