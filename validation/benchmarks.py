"""
Benchmark definitions.

Each benchmark is an *analysis* dict — exactly what the LLM produces for a
problem description — so the validation exercises the real generator path.
After the run, ``evaluate`` reads the results with :mod:`app.postprocess`
(pure Python, no OpenFOAM function objects) and returns metrics with their
reference values, and ``plot`` draws a comparison figure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from app.postprocess import FoamCase
from app.postprocess.render import plot_cells


@dataclass
class Metric:
    name: str
    value: float
    reference: float
    tolerance: float          # relative, e.g. 0.02 = 2 %
    unit: str = ""
    source: str = ""

    @property
    def error(self) -> float:
        return (self.value - self.reference) / self.reference

    @property
    def passed(self) -> bool:
        return bool(abs(self.error) <= self.tolerance)


@dataclass
class Benchmark:
    key: str
    title: str
    description: str          # the plain-English problem, as a user would type it
    reference: str            # citation for the reference data
    solver: str
    analysis: dict
    end_time: float           # iterations (steady) or seconds (transient)
    evaluate: Callable[[FoamCase, dict], list[Metric]]
    plot: Callable[[FoamCase, dict, list[Metric], str], None]
    notes: str = ""
    meta: dict = field(default_factory=dict)
    # (interval, count): save every `interval` and keep the last `count` results — for
    # time-series metrics, since OpenFOAM function objects (probes) crash in this build
    sampling: tuple[float, int] | None = None
    max_cores: int = 1         # run in parallel on up to this many cores


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _column(case: FoamCase, x_target: float) -> np.ndarray:
    """Indices of the cells in the mesh column whose centre x is closest to *x_target*."""
    x = case.cell_centres[:, 0]
    xs = np.unique(np.round(x, 10))
    x_col = xs[np.argmin(np.abs(xs - x_target))]
    return np.where(np.isclose(x, x_col, rtol=0, atol=1e-9 + 1e-6 * abs(x_col)))[0]


def _row(case: FoamCase, y_target: float, coord=None) -> np.ndarray:
    y = case.cell_centres[:, 1] if coord is None else coord
    ys = np.unique(np.round(y, 10))
    y_row = ys[np.argmin(np.abs(ys - y_target))]
    return np.where(np.isclose(y, y_row, rtol=0, atol=1e-9 + 1e-6 * abs(y_row)))[0]


def _analysis(solver: str, geometry: str, *, nu: float, U: float, L: float,
              turb: str = "laminar", fluid: str = "air", rho: float = 1.2,
              bcs: list | None = None, geometry_params: dict | None = None,
              mesh: dict | None = None, **extra) -> dict:
    a = {
        "solver": solver,
        "solver_justification": "Validation benchmark.",
        "fluid": {"name": fluid, "nu": nu, "rho": rho},
        "turb_model": turb,
        "geometry_type": geometry,
        "geometry_params": geometry_params or {},
        "boundary_conditions": bcs or [
            {"patch_name": "inlet", "type": "fixedValue", "values": {"U": [U, 0.0, 0.0]}},
            {"patch_name": "outlet", "type": "zeroGradient", "values": {}},
            {"patch_name": "walls", "type": "noSlip", "values": {}},
        ],
        "numerical_schemes": {},
        "Re": U * L / nu if U else None,
        "is_transient": solver in ("pimpleFoam", "interFoam"),
        "suggested_mesh": mesh or {},
        "characteristic_length": L,
        "characteristic_velocity": U,
        "warnings": [],
        "n_cores": 1,
    }
    a.update(extra)
    return a


def _fig(nrows=1, ncols=2, size=(11, 4)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(nrows, ncols, figsize=size)
    return plt, fig, axes


# ─────────────────────────────────────────────────────────────────────────────
# 1. Plane Poiseuille flow (laminar channel) — analytical solution
# ─────────────────────────────────────────────────────────────────────────────

_CH = {"H": 0.1, "L": 2.0, "U": 0.01, "nu": 1e-5}          # Re_H = 100


def _channel_eval(case: FoamCase, b: dict) -> list[Metric]:
    t = case.latest_time()
    U, p = case.field("U", t), case.field("p", t)
    H, L, Um, nu = b["H"], b["L"], b["U"], b["nu"]
    col = _column(case, 0.8 * L)
    u_max = U[col, 0].max()
    row = _row(case, H / 2)
    xr = case.cell_centres[row, 0]
    sel = row[(xr > 0.5 * L) & (xr < 0.9 * L)]
    dpdx = np.polyfit(case.cell_centres[sel, 0], p[sel], 1)[0]
    y = case.cell_centres[col, 1]
    u_an = 6 * Um * (y / H) * (1 - y / H)
    src = "analytical (plane Poiseuille)"
    return [
        Metric("Centreline velocity u_max / U_mean", u_max / Um, 1.5, 0.01, "", src),
        Metric("Pressure gradient dp/dx", dpdx, -12 * nu * Um / H**2, 0.02, "m/s²", src),
        Metric("Profile: 1 + max deviation from exact / u_max", 1 + np.abs(U[col, 0] - u_an).max() / (1.5 * Um),
               1.0, 0.01, "(1 + error)", src),
    ]


def _channel_plot(case: FoamCase, b: dict, metrics: list[Metric], out: str) -> None:
    t = case.latest_time()
    U, p = case.field("U", t), case.field("p", t)
    H, L, Um, nu = b["H"], b["L"], b["U"], b["nu"]
    plt, fig, (a1, a2) = _fig()
    col = _column(case, 0.8 * L)
    y = case.cell_centres[col, 1]
    order = np.argsort(y)
    yy = np.linspace(0, H, 200)
    a1.plot(6 * (yy / H) * (1 - yy / H), yy / H, "k-", label="analytical")
    a1.plot(U[col, 0][order] / Um, y[order] / H, "o", ms=4, mfc="none", color="C0", label="generated case")
    a1.set(xlabel="u / U_mean", ylabel="y / H", title="Velocity profile at x = 0.8 L")
    a1.legend()
    row = _row(case, H / 2)
    x = case.cell_centres[row, 0]
    o = np.argsort(x)
    a2.plot(x[o] / L, p[row][o], "-", color="C0", label="generated case")
    xs = np.linspace(0.4, 1.0, 50) * L
    dp = -12 * nu * Um / H**2
    a2.plot(xs / L, dp * (xs - L), "k--", label="analytical slope (developed flow)")
    a2.set(xlabel="x / L", ylabel="p (kinematic) [m²/s²]", title="Centreline pressure")
    a2.legend()
    fig.suptitle("Plane Poiseuille flow, Re_H = 100 (simpleFoam, laminar)")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


POISEUILLE_CHANNEL = Benchmark(
    key="poiseuille_channel",
    title="Plane Poiseuille flow (laminar channel)",
    description="Laminar flow of a fluid with ν = 1e-5 m²/s between two parallel plates 0.1 m "
                "apart, 2 m long, mean velocity 0.01 m/s (Re = 100).",
    reference="Analytical: u(y) = 6·U·(y/H)(1 − y/H), dp/dx = −12 ν U / H²",
    solver="simpleFoam",
    analysis=_analysis("simpleFoam", "channel", nu=_CH["nu"], U=_CH["U"], L=_CH["H"],
                       mesh={"domain_x": _CH["L"], "domain_y": _CH["H"], "nx": 200, "ny": 40}),
    end_time=3000,
    evaluate=lambda c, _a: _channel_eval(c, _CH),
    plot=lambda c, _a, m, out: _channel_plot(c, _CH, m, out),
    meta=_CH,
)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Hagen–Poiseuille flow (laminar pipe, axisymmetric wedge) — analytical
# ─────────────────────────────────────────────────────────────────────────────

_PI = {"R": 0.05, "L": 2.0, "U": 0.01, "nu": 1e-5}          # Re_D = 100


def _pipe_radius(case: FoamCase) -> np.ndarray:
    cc = case.cell_centres
    return np.hypot(cc[:, 1], cc[:, 2])


def _pipe_eval(case: FoamCase, b: dict) -> list[Metric]:
    t = case.latest_time()
    U, p = case.field("U", t), case.field("p", t)
    R, L, Um, nu = b["R"], b["L"], b["U"], b["nu"]
    col = _column(case, 0.8 * L)
    u_max = U[col, 0].max()
    r = _pipe_radius(case)
    row = _row(case, 0.0, coord=r)
    xr = case.cell_centres[row, 0]
    sel = row[(xr > 0.5 * L) & (xr < 0.9 * L)]
    dpdx = np.polyfit(case.cell_centres[sel, 0], p[sel], 1)[0]
    u_an = 2 * Um * (1 - (r[col] / R) ** 2)
    src = "analytical (Hagen–Poiseuille)"
    return [
        Metric("Centreline velocity u_max / U_mean", u_max / Um, 2.0, 0.01, "", src),
        Metric("Pressure gradient dp/dx", dpdx, -8 * nu * Um / R**2, 0.02, "m/s²", src),
        Metric("Profile: 1 + max deviation from exact / u_max", 1 + np.abs(U[col, 0] - u_an).max() / (2 * Um),
               1.0, 0.015, "(1 + error)", src),
    ]


def _pipe_plot(case: FoamCase, b: dict, metrics: list[Metric], out: str) -> None:
    t = case.latest_time()
    U, p = case.field("U", t), case.field("p", t)
    R, L, Um, nu = b["R"], b["L"], b["U"], b["nu"]
    plt, fig, (a1, a2) = _fig()
    col = _column(case, 0.8 * L)
    r = _pipe_radius(case)[col]
    o = np.argsort(r)
    rr = np.linspace(0, R, 200)
    a1.plot(2 * (1 - (rr / R) ** 2), rr / R, "k-", label="analytical")
    a1.plot(U[col, 0][o] / Um, r[o] / R, "o", ms=3, mfc="none", color="C1", label="generated case")
    a1.set(xlabel="u / U_mean", ylabel="r / R", title="Velocity profile at x = 0.8 L")
    a1.legend()
    plot_cells(a2, case, U[:, 0] / Um, cmap="viridis", label="u / U_mean")
    a2.set_aspect("auto")
    a2.set(xlabel="x [m]", ylabel="r [m]", title="Axial velocity (5° wedge mesh)")
    fig.suptitle("Hagen–Poiseuille pipe flow, Re_D = 100 (simpleFoam, laminar, axisymmetric)")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


POISEUILLE_PIPE = Benchmark(
    key="poiseuille_pipe",
    title="Hagen–Poiseuille flow (laminar pipe, axisymmetric)",
    description="Laminar flow in a circular pipe of radius 5 cm and length 2 m, mean velocity "
                "0.01 m/s, ν = 1e-5 m²/s (Re_D = 100).",
    reference="Analytical: u(r) = 2·U·(1 − r²/R²), dp/dx = −8 ν U / R²",
    solver="simpleFoam",
    analysis=_analysis("simpleFoam", "pipe", nu=_PI["nu"], U=_PI["U"], L=2 * _PI["R"],
                       geometry_params={"pipe_radius": _PI["R"], "pipe_length": _PI["L"]}),
    end_time=3000,
    evaluate=lambda c, _a: _pipe_eval(c, _PI),
    plot=lambda c, _a, m, out: _pipe_plot(c, _PI, m, out),
    meta=_PI,
)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Differentially heated square cavity, Ra = 1e5 — de Vahl Davis (1983)
# ─────────────────────────────────────────────────────────────────────────────

def _cavity_setup(Ra: float = 1e5, dT: float = 10.0, T_mean: float = 300.0,
                  nu: float = 1.5e-5, Pr: float = 0.713) -> dict:
    g, beta = 9.81, 1.0 / T_mean
    L = (Ra * nu**2 / (g * beta * dT * Pr)) ** (1.0 / 3.0)
    return {"Ra": Ra, "dT": dT, "T_hot": T_mean + dT / 2, "T_cold": T_mean - dT / 2,
            "nu": nu, "Pr": Pr, "L": L, "alpha": nu / Pr}


_CAV = _cavity_setup()

# de Vahl Davis, "Natural convection of air in a square cavity: a bench mark numerical
# solution", Int. J. Numer. Meth. Fluids 3 (1983) 249–264 — Ra = 1e5 values.
_DVD = {"Nu_avg": 4.519, "u_max": 34.73, "y_umax": 0.855, "v_max": 68.59, "x_vmax": 0.066}


def _rows_by_y(case: FoamCase) -> list[np.ndarray]:
    y = np.round(case.cell_centres[:, 1], 10)
    return [np.where(y == v)[0] for v in np.unique(y)]


def _nusselt(case: FoamCase, b: dict, t: str) -> np.ndarray:
    """Local Nu along the hot wall (x = 0) from a 2nd-order one-sided T gradient."""
    T = case.field("T", t)
    x = case.cell_centres[:, 0]
    nus = []
    for row in _rows_by_y(case):
        o = row[np.argsort(x[row])][:2]
        coeff = np.polyfit([0.0, x[o[0]], x[o[1]]], [b["T_hot"], T[o[0]], T[o[1]]], 2)
        nus.append(-coeff[1] * b["L"] / b["dT"])
    return np.array(nus)


def _centreline(case: FoamCase, U: np.ndarray, axis: int, b: dict):
    """Velocity on the vertical (axis=0 → u vs y) or horizontal (axis=1 → v vs x) centreline,
    averaged over the two cell columns/rows either side of the centre."""
    cc = case.cell_centres
    coord = cc[:, axis]
    vals = np.unique(np.round(coord, 10))
    mid = vals[np.argsort(np.abs(vals - b["L"] / 2))[:2]]
    other = 1 - axis
    lines = []
    for m in mid:
        idx = np.where(np.isclose(coord, m, atol=1e-12))[0]
        idx = idx[np.argsort(cc[idx, other])]
        lines.append((cc[idx, other], U[idx, axis]))   # u on vertical, v on horizontal line
    pos = lines[0][0]
    vel = 0.5 * (lines[0][1] + lines[1][1])
    return pos / b["L"], vel * b["L"] / b["alpha"]


def _cavity_eval(case: FoamCase, b: dict) -> list[Metric]:
    t = case.latest_time()
    U = case.field("U", t)
    nu_avg = _nusselt(case, b, t).mean()
    _, u = _centreline(case, U, 0, b)
    _, v = _centreline(case, U, 1, b)
    src = "de Vahl Davis (1983)"
    return [
        Metric("Average Nusselt number (hot wall)", nu_avg, _DVD["Nu_avg"], 0.03, "", src),
        Metric("Max u on vertical centreline · L/α", u.max(), _DVD["u_max"], 0.05, "", src),
        Metric("Max v on horizontal centreline · L/α", v.max(), _DVD["v_max"], 0.05, "", src),
    ]


def _cavity_plot(case: FoamCase, b: dict, metrics: list[Metric], out: str) -> None:
    t = case.latest_time()
    T, U = case.field("T", t), case.field("U", t)
    plt, fig, (a1, a2, a3) = _fig(1, 3, (15, 4.4))
    plot_cells(a1, case, T, cmap="coolwarm", label="T [K]")
    cc = case.cell_centres
    step = max(1, int(math.sqrt(case.n_cells) // 20))
    grid = np.round(cc[:, 0] / (b["L"] / math.sqrt(case.n_cells))).astype(int)
    gridy = np.round(cc[:, 1] / (b["L"] / math.sqrt(case.n_cells))).astype(int)
    sel = (grid % step == 0) & (gridy % step == 0)
    a1.quiver(cc[sel, 0], cc[sel, 1], U[sel, 0], U[sel, 1], color="k", scale_units="xy",
              angles="xy", width=0.004)
    a1.set(title="Temperature and velocity", xlabel="x [m]", ylabel="y [m]")

    y, u = _centreline(case, U, 0, b)
    a2.plot(u, y, "-", color="C0", label="generated case")
    a2.plot(_DVD["u_max"], _DVD["y_umax"], "ks", label="de Vahl Davis u_max")
    a2.set(xlabel="u · L / α", ylabel="y / L", title="u on vertical centreline")
    a2.legend(loc="lower right")

    nus = _nusselt(case, b, t)
    yy = np.unique(np.round(cc[:, 1], 10)) / b["L"]
    a3.plot(nus, yy, "-", color="C3", label=f"local Nu (mean {nus.mean():.3f})")
    a3.axvline(_DVD["Nu_avg"], color="k", ls="--", label=f"de Vahl Davis mean {_DVD['Nu_avg']}")
    a3.set(xlabel="Nu", ylabel="y / L", title="Hot-wall Nusselt number")
    a3.legend()
    fig.suptitle("Differentially heated cavity, Ra = 10⁵ (buoyantSimpleFoam, laminar)")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


NATURAL_CONVECTION_CAVITY = Benchmark(
    key="cavity_ra1e5",
    title="Natural convection in a square cavity, Ra = 10⁵",
    description=f"Natural convection of air in a square cavity of side {_CAV['L'] * 100:.2f} cm, "
                f"hot left wall at {_CAV['T_hot']:.0f} K, cold right wall at {_CAV['T_cold']:.0f} K, "
                "adiabatic top and bottom (Ra = 1e5). Steady state, laminar.",
    reference="de Vahl Davis, Int. J. Numer. Meth. Fluids 3 (1983) 249–264",
    solver="buoyantSimpleFoam",
    analysis=_analysis(
        "buoyantSimpleFoam", "cavity", nu=_CAV["nu"], U=0.0, L=_CAV["L"],
        bcs=[{"patch_name": "hotWall", "type": "fixedValue", "values": {"T": _CAV["T_hot"]}},
             {"patch_name": "coldWall", "type": "fixedValue", "values": {"T": _CAV["T_cold"]}}],
        mesh={"refinement": 2.0},
    ),
    end_time=15000,           # stops earlier once residualControl is met (~9 000 iterations)
    evaluate=lambda c, _a: _cavity_eval(c, _CAV),
    plot=lambda c, _a, m, out: _cavity_plot(c, _CAV, m, out),
    notes="Compressible perfect-gas solver with ΔT = 10 K (≈ Boussinesq); uniform 100 × 100 mesh; "
          "the reference is a Boussinesq solution at Pr = 0.71 (case: Pr = 0.713).",
    meta=_CAV,
)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Vortex shedding behind a circular cylinder, Re = 100 — Williamson (1996)
# ─────────────────────────────────────────────────────────────────────────────

_CYL = {"D": 1.0, "U": 1.0, "nu": 0.01, "L": 30.0, "H": 20.0, "cx": 10.0, "Re": 100.0}
# Williamson, Annu. Rev. Fluid Mech. 28 (1996) 477–539: St = 0.2665 − 1.018/√Re (laminar shedding)
_ST_REF = 0.2665 - 1.018 / math.sqrt(_CYL["Re"])


def _wake_probe_series(case: FoamCase, b: dict) -> tuple[np.ndarray, np.ndarray]:
    """Cross-stream velocity u_y at (cx + 2D, cy) for every saved time. code_saturne results
    come from its monitoring probe at that point (every time step), over the same last
    50 D/U as the OpenFOAM samples."""
    if hasattr(case, "probe_series"):
        t, uy = case.probe_series("Velocity[Y]")
        keep = t >= t.max() - 50.0 * b["D"] / b["U"]
        return t[keep], uy[keep]
    cc = case.cell_centres
    target = np.array([b["cx"] + 2 * b["D"], b["H"] / 2])
    cell = int(np.argmin(np.hypot(cc[:, 0] - target[0], cc[:, 1] - target[1])))
    times = [t for t in case.times() if float(t) > 0]
    return np.array([float(t) for t in times]), np.array([case.field("U", t)[cell, 1] for t in times])


def _shedding_period(t: np.ndarray, uy: np.ndarray) -> tuple[float, int]:
    """Mean period from upward zero crossings of u_y − mean (linearly interpolated)."""
    s = uy - uy.mean()
    idx = np.where((s[:-1] < 0) & (s[1:] >= 0))[0]
    crossings = t[idx] - s[idx] * (t[idx + 1] - t[idx]) / (s[idx + 1] - s[idx])
    if len(crossings) < 3:
        return float("nan"), 0
    return float(np.diff(crossings).mean()), len(crossings) - 1


def _cylinder_eval(case: FoamCase, b: dict) -> list[Metric]:
    t, uy = _wake_probe_series(case, b)
    period, n = _shedding_period(t, uy)
    st = b["D"] / (b["U"] * period) if n else 0.0
    return [Metric(f"Strouhal number St = fD/U ({n} periods)", st, _ST_REF, 0.05, "",
                   "Williamson (1996)")]


def _cylinder_plot(case: FoamCase, b: dict, metrics: list[Metric], out: str) -> None:
    plt, fig, axes = _fig(1, 3, (16, 4.2))
    a1, a2, a3 = axes
    tl = case.latest_time()
    U = case.field("U", tl)
    plot_cells(a1, case, U[:, 1] / b["U"], cmap="RdBu_r", vmin=-0.8, vmax=0.8, label="u_y / U")
    a1.plot(b["cx"] + 2 * b["D"], b["H"] / 2, "k+", ms=10, mew=2)
    a1.set(xlim=(b["cx"] - 3, b["cx"] + 15), ylim=(b["H"] / 2 - 5, b["H"] / 2 + 5),
           title=f"u_y at t = {float(tl):.0f} D/U  (+ probe)", xlabel="x / D", ylabel="y / D")

    t, uy = _wake_probe_series(case, b)
    a2.plot(t, uy / b["U"], "-", color="C0")
    a2.set(xlabel="t · U / D", ylabel="u_y / U at probe", title="Wake velocity at x = 2D behind the cylinder")

    re = np.linspace(50, 180, 100)
    a3.plot(re, 0.2665 - 1.018 / np.sqrt(re), "k-", label="Williamson (1996) fit")
    a3.plot(b["Re"], metrics[0].value, "o", ms=9, color="C3", label=f"generated case: St = {metrics[0].value:.4f}")
    a3.set(xlabel="Re", ylabel="St", title="Strouhal number")
    a3.legend()
    fig.suptitle("Laminar vortex shedding behind a cylinder, Re = 100 (pimpleFoam)")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


CYLINDER_SHEDDING = Benchmark(
    key="cylinder_re100",
    title="Vortex shedding behind a circular cylinder, Re = 100",
    description="Laminar flow past a 2-D circular cylinder at Re = 100 (D = 1 m, U = 1 m/s, "
                "ν = 0.01 m²/s), in a domain 30 D long and 20 D high with the cylinder 10 D from "
                "the inlet. Transient; measure the vortex-shedding frequency.",
    reference="Williamson, Annu. Rev. Fluid Mech. 28 (1996): St = 0.2665 − 1.018/√Re",
    solver="pimpleFoam",
    analysis=_analysis(
        "pimpleFoam", "cylinder", nu=_CYL["nu"], U=_CYL["U"], L=_CYL["D"],
        bcs=[{"patch_name": "inlet", "type": "fixedValue", "values": {"U": [_CYL["U"], 0.0, 0.0]}},
             {"patch_name": "outlet", "type": "zeroGradient", "values": {}}],
        geometry_params={"cylinder_radius": _CYL["D"] / 2, "channel_length": _CYL["L"],
                         "channel_height": _CYL["H"], "cx": _CYL["cx"], "cy": _CYL["H"] / 2},
    ),
    # From a symmetric start the wake needs ~100 D/U to start shedding and more to saturate:
    # sampling while the amplitude still grows under-predicted St by 4 %.
    end_time=200.0,
    sampling=(0.25, 200),           # last 50 D/U ≈ 8 periods, 24 samples per period
    max_cores=6,
    evaluate=lambda c, _a: _cylinder_eval(c, _CYL),
    plot=lambda c, _a, m, out: _cylinder_plot(c, _CYL, m, out),
    notes="Symmetry planes 10 D above and below the cylinder (5 % blockage) raise St slightly "
          "compared with an unbounded flow. The period is measured from zero crossings of u_y "
          "at a probe 2 D behind the cylinder.",
    meta=_CYL,
)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Turbulent backward-facing step — Driver & Seegmiller (1985)
# ─────────────────────────────────────────────────────────────────────────────

_BS = {"H": 0.0127, "U": 44.2, "nu": 1.5e-5}
_BS.update(xs=1.0,                      # ≈ 80 H of inlet channel: δ ≈ 1.5 H at the step, as measured
           height=9 * _BS["H"],         # expansion ratio 9/8 (inlet channel 8 H high)
           L=1.0 + 30 * _BS["H"])       # 30 H downstream of the step
_XR_REF = 6.26                          # measured reattachment length x_r / H (± 0.10)


def _floor_ux(case: FoamCase, b: dict) -> tuple[np.ndarray, np.ndarray]:
    """u_x in the wall-adjacent cell row on the floor downstream of the step."""
    cc = case.cell_centres
    U = case.field("U", case.latest_time())
    down = cc[:, 0] > b["xs"]
    y_first = cc[down, 1].min()
    row = np.where(down & np.isclose(cc[:, 1], y_first, rtol=0, atol=1e-9))[0]
    order = np.argsort(cc[row, 0])
    return cc[row, 0][order], U[row, 0][order]


def _reattachment(x: np.ndarray, ux: np.ndarray, b: dict) -> float:
    """First x behind the recirculation where the floor velocity turns positive, as x_r / H."""
    neg = np.where(ux < 0)[0]
    if not len(neg):
        return 0.0
    i = neg[-1]                         # last backward-flowing cell; the next one flows forward
    if i + 1 >= len(x):
        return float("nan")
    x0 = x[i] - ux[i] * (x[i + 1] - x[i]) / (ux[i + 1] - ux[i])
    return (x0 - b["xs"]) / b["H"]


def _backstep_eval(case: FoamCase, b: dict) -> list[Metric]:
    x, ux = _floor_ux(case, b)
    return [Metric("Reattachment length x_r / H (k-ω SST)", _reattachment(x, ux, b), _XR_REF, 0.10,
                   "", "Driver & Seegmiller (1985)")]


def _backstep_plot(case: FoamCase, b: dict, metrics: list[Metric], out: str) -> None:
    plt, fig, (a1, a2) = _fig(1, 2, (15, 4.2))
    U = case.field("U", case.latest_time())
    plot_cells(a1, case, U[:, 0] / b["U"], cmap="RdBu_r", vmin=-1.1, vmax=1.1, label="u_x / U")
    cc = case.cell_centres
    a1.tricontour(cc[:, 0], cc[:, 1], U[:, 0], levels=[0.0], colors="k", linewidths=0.8)
    a1.set(xlim=(b["xs"] - 2 * b["H"], b["xs"] + 12 * b["H"]), ylim=(0, 3 * b["H"]),
           title="Streamwise velocity behind the step (black: u_x = 0)", xlabel="x [m]", ylabel="y [m]")
    x, ux = _floor_ux(case, b)
    xr = metrics[0].value
    a2.plot((x - b["xs"]) / b["H"], ux / b["U"], "-", color="C0", label="generated case (wall cell)")
    a2.axhline(0, color="0.5", lw=0.8)
    a2.axvspan(_XR_REF - 0.10, _XR_REF + 0.10, color="0.8", label="Driver & Seegmiller: 6.26 ± 0.10")
    a2.axvline(xr, color="C3", ls="--", label=f"generated case: {xr:.2f}")
    a2.set(xlim=(0, 15), xlabel="(x − x_step) / H", ylabel="u_x / U next to the floor",
           title="Reattachment length")
    a2.legend(loc="lower right")
    fig.suptitle("Turbulent backward-facing step, Re_H = 37 400 (simpleFoam, k-ω SST)")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


BACKWARD_FACING_STEP = Benchmark(
    key="backstep_driver",
    title="Turbulent backward-facing step (Driver & Seegmiller)",
    description="Air at 44.2 m/s flows over a backward-facing step 1.27 cm high; the channel is 8 step "
                "heights high upstream and 9 downstream. Steady, turbulent (k-omega SST).",
    reference="Driver & Seegmiller, AIAA J. 23 (1985) 163–171: x_r / H = 6.26 ± 0.10",
    solver="simpleFoam",
    analysis=_analysis(
        "simpleFoam", "backstep", nu=_BS["nu"], U=_BS["U"], L=_BS["H"], turb="kOmegaSST",
        geometry_params={"channel_length": _BS["L"], "channel_height": _BS["height"],
                         "step_height": _BS["H"], "step_x": _BS["xs"]},
        mesh={"refinement": 3.0},
    ),
    end_time=4000,
    max_cores=6,
    evaluate=lambda c, _a: _backstep_eval(c, _BS),
    plot=lambda c, _a, m, out: _backstep_plot(c, _BS, m, out),
    notes="RANS models predict reattachment less precisely than laminar flows (k-ε typically "
          "under-predicts it by 15–20 %), hence the ±10 % tolerance. Uniform inflow developed over "
          "≈ 80 H of channel; wall functions on wall-graded cells. Mesh sensitivity: 23 600 → 42 100 "
          "cells changes x_r / H by 1.3 % (5.96 → 6.04), so the deviation comes from the turbulence "
          "model, not the mesh.",
    meta=_BS,
)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Steady wake behind a circular cylinder, Re = 40 — recirculation length
# ─────────────────────────────────────────────────────────────────────────────

_C40 = {"D": 1.0, "U": 1.0, "nu": 0.025, "L": 30.0, "H": 20.0, "cx": 10.0, "Re": 40.0}
# Wake (recirculation) length behind the cylinder, L_w/D at Re = 40: 2.35 (Dennis & Chang 1970),
# 2.24 (Fornberg 1980), 2.13 (Coutanceau & Bouard 1977, experiment). Reference: their mean.
_LW_REF = (2.35 + 2.24 + 2.13) / 3


def _wake_length(case, b: dict) -> tuple[float, np.ndarray, np.ndarray]:
    """L_w/D from u_x along the wake centreline: first sign change behind the cylinder."""
    t = case.latest_time()
    U = case.field("U", t)
    cc = case.cell_centres
    D, cx, cy = b["D"], b["cx"], b["H"] / 2
    band = (np.abs(cc[:, 1] - cy) < 0.5 * D) & (cc[:, 0] > cx + D / 2) & (cc[:, 0] < cx + 8 * D)
    x, y, ux = cc[band, 0], cc[band, 1], U[band, 0]
    # at each streamwise position, average the two cells closest to the centreline
    xs, us = [], []
    for xv in np.unique(np.round(x, 4)):
        m = np.where(np.isclose(x, xv, atol=1e-4))[0]
        near = m[np.argsort(np.abs(y[m] - cy))[:2]]
        xs.append(xv); us.append(ux[near].mean())
    xs, us = np.array(xs), np.array(us)
    if not np.any(us > 0):
        return float("nan"), xs, us
    # end of the backflow region: the crossing after the last negative u_x (cells right
    # behind the cylinder, off the centreline in the O-grid, can show spurious positives)
    neg = np.where(us < 0)[0]
    if not len(neg) or neg[-1] == len(us) - 1:
        return float("nan"), xs, us
    i = int(neg[-1]) + 1
    x0 = xs[i - 1] - us[i - 1] * (xs[i] - xs[i - 1]) / (us[i] - us[i - 1])
    return (x0 - (cx + D / 2)) / D, xs, us


def _c40_eval(case, b: dict) -> list[Metric]:
    lw, _, _ = _wake_length(case, b)
    return [Metric("Recirculation length L_w / D", lw, _LW_REF, 0.10, "",
                   "Dennis & Chang 1970, Fornberg 1980, Coutanceau & Bouard 1977 (mean 2.24)")]


def _c40_plot(case, b: dict, metrics: list[Metric], out: str) -> None:
    plt, fig, ax = _fig(1, 1, (7, 3.5))
    lw, xs, us = _wake_length(case, b)
    D = b["D"]
    ax.plot((xs - b["cx"] - D / 2) / D, us / b["U"], "o-", ms=3, label="generated case")
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(_LW_REF, color="tab:red", ls="--", label=f"reference L_w/D = {_LW_REF:.2f}")
    ax.set_xlabel("distance behind the cylinder / D"); ax.set_ylabel("u_x / U on the centreline")
    ax.set_title(f"Re = 40: L_w/D = {lw:.3f}"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


CYLINDER_RE40 = Benchmark(
    key="cylinder_re40",
    title="Steady wake behind a circular cylinder, Re = 40",
    description="Steady laminar flow past a 2-D circular cylinder at Re = 40 (D = 1 m, U = 1 m/s, "
                "ν = 0.025 m²/s), in a domain 30 D long and 20 D high with the cylinder 10 D from "
                "the inlet. Measure the length of the recirculation bubble behind it.",
    reference="Dennis & Chang, J. Fluid Mech. 42 (1970); Fornberg, J. Fluid Mech. 98 (1980); "
              "Coutanceau & Bouard, J. Fluid Mech. 79 (1977): L_w/D = 2.13–2.35",
    solver="simpleFoam",
    analysis=_analysis(
        "simpleFoam", "cylinder", nu=_C40["nu"], U=_C40["U"], L=_C40["D"],
        bcs=[{"patch_name": "inlet", "type": "fixedValue", "values": {"U": [_C40["U"], 0.0, 0.0]}},
             {"patch_name": "outlet", "type": "zeroGradient", "values": {}}],
        geometry_params={"cylinder_radius": _C40["D"] / 2, "channel_length": _C40["L"],
                         "channel_height": _C40["H"], "cx": _C40["cx"], "cy": _C40["H"] / 2},
    ),
    end_time=3000,
    evaluate=lambda c, _a: _c40_eval(c, _C40),
    plot=lambda c, _a, m, out: _c40_plot(c, _C40, m, out),
    notes="Below the shedding threshold (Re ≈ 47) the wake is steady. The literature spread "
          "(2.13–2.35) sets the ±10 % tolerance; symmetry planes 10 D away (5 % blockage). "
          "Also run with code_saturne (validation/compare.py).",
    meta=_C40,
)


BENCHMARKS: dict[str, Benchmark] = {
    b.key: b for b in (POISEUILLE_CHANNEL, POISEUILLE_PIPE, NATURAL_CONVECTION_CAVITY,
                       CYLINDER_SHEDDING, BACKWARD_FACING_STEP, CYLINDER_RE40)
}
