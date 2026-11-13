# LLM evaluation

How accurately each model turns a plain-English description into a CFD setup, on 20 cases with known answers ([`evals/cases.json`](../../evals/cases.json)).
**Raw**: the model's answer as returned. **Sanitised**: after the validation layer (`app/sanitize.py`), i.e. what the app actually generates.

| Model | Cases | Solver | Geometry | Laminar/RANS | Steady/transient | Re (±15 %) | Viscosity | **Fully correct** | Median time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `llama3.1:8b` (ollama) | 20 | 60 % | 80 % | 69 % | 62 % | 75 % | 100 % | **30 %** | 60 s |
| ↳ after sanitiser | 20 | 75 % | 80 % | 100 % | 88 % | 94 % | 100 % | **60 %** |  |
| `mistral` (ollama) | 20 | 80 % | 85 % | 75 % | 75 % | 56 % | 100 % | **45 %** | 73 s |
| ↳ after sanitiser | 20 | 90 % | 85 % | 94 % | 88 % | 75 % | 100 % | **70 %** |  |
| `qwen2.5:7b` (ollama) | 20 | 75 % | 85 % | 50 % | 62 % | 81 % | 100 % | **30 %** | 66 s |
| ↳ after sanitiser | 20 | 95 % | 85 % | 100 % | 75 % | 100 % | 100 % | **80 %** |  |

## Per case (sanitised: ✅ fully correct, ❌ wrong fields listed)

| Case | `llama3.1:8b` | `mistral` | `qwen2.5:7b` |
|---|---|---|---|
| cyl_re100 | ✅ | ✅ | ✅ |
| cyl_re40 | ✅ | ✅ | ✅ |
| cyl_air_turbulent | ❌ solver | ✅ | ✅ |
| pipe_turbulent | ✅ | ✅ | ✅ |
| pipe_oil_laminar | ❌ solver | ❌ Re | ✅ |
| plates_laminar | ❌ Re | ❌ laminar, Re | ✅ |
| duct_turbulent | ✅ | ✅ | ✅ |
| channel_startup | ✅ | ✅ | ✅ |
| backstep | ❌ geometry | ✅ | ❌ geometry |
| t_junction | ✅ | ❌ Re | ✅ |
| building_wind | ❌ solver, geometry | ✅ | ✅ |
| square_shedding | ❌ geometry | ✅ | ❌ geometry |
| wing_low_speed | ❌ solver | ❌ solver, geometry, Re | ✅ |
| airfoil_transonic | ✅ | ❌ geometry | ✅ |
| channel_compressible | ✅ | ✅ | ✅ |
| cavity_temperatures | ✅ | ✅ | ✅ |
| cavity_ra1e5 | ✅ | ✅ | ✅ |
| dam_break | ❌ solver, geometry, transient | ❌ solver, geometry, transient | ❌ solver, geometry, transient |
| dam_sloshing | ✅ | ✅ | ❌ transient |
| heated_channel_forced | ✅ | ✅ | ✅ |

## Notes

- These 20 cases were also used to find weaknesses in the sanitiser (solver/geometry conflicts, laminar vs RANS by Reynolds number, Mach > 0.3 needs a compressible solver, a stated Mach number sets the velocity). The rules are general physics, not tuned to particular sentences, but the *after sanitiser* numbers are measured on the same cases; a fresh set of descriptions would give an unbiased estimate.
- Not fixed on purpose: all local models misread the dam-break description (they used gravity, 9.81, as an inlet velocity). Catching that would need keyword rules fitted to these test sentences, which would inflate the score without improving the app.
- Local models ran on a laptop GPU (GTX 1650 Ti); cloud models are much faster.
