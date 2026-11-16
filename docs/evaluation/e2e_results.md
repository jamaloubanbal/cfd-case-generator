# End-to-end benchmark

From a plain-English description to a judged OpenFOAM result, with no human in the loop: the model's answer is sanitised, generated, run with the rule-based repair agent and judged by the [result reviewer](../GUIDE.md#result-review). Re-run with `python -m evals.e2e`.

## gemini__gemini-3.1-flash-lite

**25 of 30 descriptions give a trustworthy result** (4 questionable, 1 unreliable, 0 did not run, 0 rejected).

| Outcome | Dev (20) | Held-out (10) |
|---|---|---|
| trustworthy | 17 | 8 |
| questionable | 2 | 2 |
| unreliable | 1 | 0 |

| Case | Set | Solver · geometry · model | Fixes | Outcome | Issues |
|---|---|---|---|---|---|
| `backstep` | dev | simpleFoam · backstep · kOmegaSST | – | **trustworthy** | – |
| `cavity_ra1e5` | dev | buoyantSimpleFoam · cavity · laminar | – | **trustworthy** | – |
| `cavity_temperatures` | dev | buoyantSimpleFoam · cavity · laminar | – | **trustworthy** | – |
| `channel_compressible` | dev | rhoSimpleFoam · channel · kOmegaSST | – | **trustworthy** | – |
| `channel_startup` | dev | pimpleFoam · channel · kEpsilon | – | **trustworthy** | – |
| `cyl_air_turbulent` | dev | pimpleFoam · cylinder · kOmegaSST | – | **trustworthy** | – |
| `cyl_re100` | dev | pimpleFoam · cylinder · laminar | – | **trustworthy** | – |
| `cyl_re40` | dev | simpleFoam · cylinder · laminar | – | **trustworthy** | – |
| `dam_break` | dev | interFoam · dambreak · laminar | – | **trustworthy** | – |
| `dam_sloshing` | dev | interFoam · dambreak · laminar | – | **trustworthy** | – |
| `duct_turbulent` | dev | simpleFoam · channel · kEpsilon | – | **trustworthy** | – |
| `h_chimney_wind` | held-out | pimpleFoam · cylinder · kOmegaSST | – | **trustworthy** | – |
| `h_fast_duct` | held-out | rhoSimpleFoam · channel · kEpsilon | – | **trustworthy** | – |
| `h_gate_release` | held-out | interFoam · dambreak · laminar | – | **trustworthy** | – |
| `h_glycerine_pipe` | held-out | simpleFoam · pipe · laminar | – | **trustworthy** | – |
| `h_microchannel` | held-out | simpleFoam · channel · laminar | – | **trustworthy** | – |
| `h_slow_wing` | held-out | simpleFoam · airfoil · kOmegaSST | – | **trustworthy** | – |
| `h_step_water` | held-out | simpleFoam · backstep · kOmegaSST | – | **trustworthy** | – |
| `h_wire_wake` | held-out | pimpleFoam · cylinder · laminar | – | **trustworthy** | – |
| `heated_channel_forced` | dev | simpleFoam · channel · kEpsilon | – | **trustworthy** | – |
| `pipe_oil_laminar` | dev | simpleFoam · pipe · laminar | – | **trustworthy** | – |
| `pipe_turbulent` | dev | simpleFoam · pipe · kEpsilon | – | **trustworthy** | – |
| `plates_laminar` | dev | simpleFoam · channel · laminar | – | **trustworthy** | – |
| `square_shedding` | dev | pimpleFoam · rectangular_obstacle · laminar | – | **trustworthy** | – |
| `t_junction` | dev | simpleFoam · t_junction · kEpsilon | – | **trustworthy** | – |
| `building_wind` | dev | pimpleFoam · rectangular_obstacle · kOmegaSST | – | **questionable** | Wall y⁺: y⁺ mean 1.54e+04, 95th percentile 8.83e+04: above 300; refine near walls. |
| `h_room_convection` | held-out | buoyantSimpleFoam · cavity · kEpsilon | – | **questionable** | Convergence: Reached endTime without converging: p_rgh residual 0.076. Run longer or check the mesh and schemes. |
| `h_wind_block` | held-out | pimpleFoam · rectangular_obstacle · kOmegaSST | – | **questionable** | Wall y⁺: y⁺ mean 608, 95th percentile 3.73e+03: above 300; refine near walls. |
| `wing_low_speed` | dev | simpleFoam · airfoil · kOmegaSST | – | **questionable** | Wall y⁺: y⁺ mean 414, 95th percentile 2.35e+03: above 300; refine near walls. |
| `airfoil_transonic` | dev | rhoSimpleFoam · airfoil · kOmegaSST | – | **unreliable** | Convergence: Not converged: p residual 0.23 at the end.<br>Wall y⁺: y⁺ mean 7.55e+03, 95th percentile 2.27e+04: above 300; refine near walls. |
