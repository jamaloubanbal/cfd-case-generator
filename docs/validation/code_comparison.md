# Code-to-code validation: OpenFOAM and code_saturne

The same benchmark (the analysis an LLM would return for its description) is generated
for both codes and measured by the same Python function. code_saturne cases come from
`app/saturne.py` (blockMesh mesh written as UNV, `setup.xml` written by code_saturne's
model layer); re-run with `python -m validation.compare`.

| Benchmark | Quantity | Reference | OpenFOAM v1912 | code_saturne 9.2 | Code-to-code gap |
|---|---|---:|---:|---:|---:|
| Plane Poiseuille flow (laminar channel) | Centreline velocity u_max / U_mean | 1.5 | 1.498 (-0.14 %) ✅ | 1.498 (-0.13 %) ✅ | +0.02 % |
| Plane Poiseuille flow (laminar channel) | Pressure gradient dp/dx | -0.00012 | -0.0001199 (-0.08 %) ✅ | -0.0001199 (-0.11 %) ✅ | -0.03 % |
| Plane Poiseuille flow (laminar channel) | Profile: 1 + max deviation from exact / u_max | 1 | – | 1.001 (+0.06 %) ✅ | – |
| Steady wake behind a circular cylinder, Re = 40 | Recirculation length L_w / D | 2.24 | 2.099 (-6.29 %) ✅ | 1.923 (-14.15 %) ❌ | -8.39 % |

Run details (code_saturne): Plane Poiseuille flow (laminar channel): 8,000 cells, 166 s, reviewer verdict trustworthy; Steady wake behind a circular cylinder, Re = 40: 20,588 cells, 818 s, reviewer verdict questionable.

**Findings.** Laminar channel flow agrees between the codes to 0.03 % and with the exact solution to 0.15 %. Behind the Re = 40 cylinder, code_saturne's recirculation is 8 % shorter than OpenFOAM's on the identical mesh: with first-order upwind it was half as long (numerical diffusion); second-order SOLU needs the local time step to converge, and its residuals level off above 1e-3 (verdict *questionable*), so the remaining gap is attributed to code_saturne's steady-state convergence on this O-grid, not settled yet.

Not yet compared: the turbulent backward-facing step (code_saturne does not converge yet with k-ω SST), the pipe (no wedge equivalent), the heated cavity and vortex shedding (natural convection and transient flow are not in the code_saturne backend yet).
