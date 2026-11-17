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
| Steady wake behind a circular cylinder, Re = 40 | Recirculation length L_w / D | 2.24 | 2.099 (-6.29 %) ✅ | 2.5 (+11.61 %) ❌ | +19.10 % |
| Vortex shedding behind a circular cylinder, Re = 100 | Strouhal number St = fD/U (7 periods) | 0.1647 | 0.1653 (+0.35 %) ✅ | 0.1652 (+0.30 %) ✅ | -0.06 % |

Run details (code_saturne): Plane Poiseuille flow (laminar channel): 8,000 cells, 154 s, reviewer verdict trustworthy; Steady wake behind a circular cylinder, Re = 40: 20,588 cells, 499 s, reviewer verdict trustworthy; Vortex shedding behind a circular cylinder, Re = 100: 20,588 cells, 692 s, reviewer verdict questionable.

**Findings.** Laminar channel flow agrees between the codes to 0.03 % and with the exact solution to 0.15 %. Vortex shedding at Re = 100 agrees to 0.06 % (St 0.1652 vs 0.1653), after switching off code_saturne's velocity slope test, which locally reverts to first-order upwind in a vortex street and had lowered St by 12 %. The steady wake at Re = 40 does not agree: code_saturne's recirculation is longer than OpenFOAM's and than the literature (2.13–2.35). Refining the mesh 2× moves OpenFOAM into the literature range (2.10 → 2.27) and code_saturne away from it (2.50 → 2.64); a time-accurate code_saturne run gives 2.45, so neither resolution nor the steady formulation explains it. Open.

Not yet compared: the turbulent backward-facing step (code_saturne does not converge yet with k-ω SST), the pipe (no wedge equivalent) and the heated cavity (natural convection is not in the code_saturne backend yet).
