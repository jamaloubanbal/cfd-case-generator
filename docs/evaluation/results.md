# LLM evaluation

How accurately each model turns a plain-English description into a CFD setup. **Raw**: the model's answer as returned. **Sanitised**: after the validation layer (`app/sanitize.py`), i.e. what the app actually generates.

## Held-out set (10 cases): unbiased estimate

Written after the sanitiser rules were developed and committed before any model was run on them ([`evals/cases_holdout.json`](../../evals/cases_holdout.json)); the sanitiser was not changed after seeing these results.

| Model | Cases | Solver | Geometry | Laminar/RANS | Steady/transient | Re (±15 %) | Viscosity | **Fully correct** | Median time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `gemini-3.1-flash-lite` (gemini) | 10 | 100 % | 100 % | 100 % | 100 % | 100 % | 100 % | **100 %** | 3 s |
| ↳ after sanitiser | 10 | 100 % | 100 % | 100 % | 100 % | 100 % | 100 % | **100 %** |  |
| `llama3.1:8b` (ollama) | 10 | 50 % | 70 % | 71 % | 50 % | 88 % | 100 % | **20 %** | 65 s |
| ↳ after sanitiser | 10 | 60 % | 70 % | 86 % | 50 % | 88 % | 100 % | **40 %** |  |
| `mistral` (ollama) | 10 | 80 % | 70 % | 57 % | 100 % | 75 % | 100 % | **40 %** | 83 s |
| ↳ after sanitiser | 10 | 90 % | 70 % | 86 % | 100 % | 75 % | 100 % | **60 %** |  |
| `qwen2.5:7b` (ollama) | 10 | 80 % | 80 % | 43 % | 50 % | 88 % | 100 % | **40 %** | 65 s |
| ↳ after sanitiser | 10 | 90 % | 80 % | 100 % | 50 % | 88 % | 100 % | **80 %** |  |

Per case, after the sanitiser (✅ fully correct, ❌ wrong fields):

| Case | `gemini-3.1-flash-lite` | `llama3.1:8b` | `mistral` | `qwen2.5:7b` |
|---|---|---|---|---|
| h_glycerine_pipe | ✅ | ✅ | ❌ geometry | ✅ |
| h_chimney_wind | ✅ | ❌ solver | ❌ geometry, Re | ✅ |
| h_wire_wake | ✅ | ❌ laminar, Re | ✅ | ✅ |
| h_fast_duct | ✅ | ✅ | ✅ | ✅ |
| h_microchannel | ✅ | ✅ | ❌ laminar, Re | ✅ |
| h_room_convection | ✅ | ✅ | ✅ | ✅ |
| h_step_water | ✅ | ❌ geometry | ✅ | ❌ geometry, Re |
| h_wind_block | ✅ | ❌ solver, geometry | ✅ | ✅ |
| h_slow_wing | ✅ | ❌ solver | ❌ solver, geometry | ✅ |
| h_gate_release | ✅ | ❌ solver, geometry, transient | ✅ | ❌ solver, geometry, transient |

## Development set (20 cases)

Used to find weaknesses in the sanitiser ([`evals/cases.json`](../../evals/cases.json)), so the *after sanitiser* numbers here are optimistic.

| Model | Cases | Solver | Geometry | Laminar/RANS | Steady/transient | Re (±15 %) | Viscosity | **Fully correct** | Median time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `gemini-3.1-flash-lite` (gemini) | 20 | 100 % | 100 % | 100 % | 100 % | 100 % | 100 % | **100 %** | 3 s |
| ↳ after sanitiser | 20 | 100 % | 100 % | 100 % | 100 % | 100 % | 100 % | **100 %** |  |
| `llama3.1:8b` (ollama) | 20 | 60 % | 80 % | 69 % | 62 % | 75 % | 100 % | **30 %** | 60 s |
| ↳ after sanitiser | 20 | 75 % | 80 % | 100 % | 88 % | 94 % | 100 % | **60 %** |  |
| `mistral` (ollama) | 20 | 80 % | 85 % | 75 % | 75 % | 56 % | 100 % | **45 %** | 73 s |
| ↳ after sanitiser | 20 | 90 % | 85 % | 94 % | 88 % | 75 % | 100 % | **70 %** |  |
| `qwen2.5:7b` (ollama) | 20 | 75 % | 85 % | 50 % | 62 % | 81 % | 100 % | **30 %** | 66 s |
| ↳ after sanitiser | 20 | 95 % | 85 % | 100 % | 75 % | 100 % | 100 % | **80 %** |  |

Per case, after the sanitiser (✅ fully correct, ❌ wrong fields):

| Case | `gemini-3.1-flash-lite` | `llama3.1:8b` | `mistral` | `qwen2.5:7b` |
|---|---|---|---|---|
| cyl_re100 | ✅ | ✅ | ✅ | ✅ |
| cyl_re40 | ✅ | ✅ | ✅ | ✅ |
| cyl_air_turbulent | ✅ | ❌ solver | ✅ | ✅ |
| pipe_turbulent | ✅ | ✅ | ✅ | ✅ |
| pipe_oil_laminar | ✅ | ❌ solver | ❌ Re | ✅ |
| plates_laminar | ✅ | ❌ Re | ❌ laminar, Re | ✅ |
| duct_turbulent | ✅ | ✅ | ✅ | ✅ |
| channel_startup | ✅ | ✅ | ✅ | ✅ |
| backstep | ✅ | ❌ geometry | ✅ | ❌ geometry |
| t_junction | ✅ | ✅ | ❌ Re | ✅ |
| building_wind | ✅ | ❌ solver, geometry | ✅ | ✅ |
| square_shedding | ✅ | ❌ geometry | ✅ | ❌ geometry |
| wing_low_speed | ✅ | ❌ solver | ❌ solver, geometry, Re | ✅ |
| airfoil_transonic | ✅ | ✅ | ❌ geometry | ✅ |
| channel_compressible | ✅ | ✅ | ✅ | ✅ |
| cavity_temperatures | ✅ | ✅ | ✅ | ✅ |
| cavity_ra1e5 | ✅ | ✅ | ✅ | ✅ |
| dam_break | ✅ | ❌ solver, geometry, transient | ❌ solver, geometry, transient | ❌ solver, geometry, transient |
| dam_sloshing | ✅ | ✅ | ✅ | ❌ transient |
| heated_channel_forced | ✅ | ✅ | ✅ | ✅ |

## Notes

- Sanitiser rules developed on the development set: solver/geometry conflicts, laminar vs RANS by Reynolds number, Ma > 0.3 needs a compressible solver, a stated Mach number sets the velocity. They are general physics, not tuned to particular sentences.
- Not fixed on purpose: all local models misread the dev-set dam-break description (they used gravity, 9.81, as an inlet velocity). Catching that would need keyword rules fitted to test sentences, which would inflate the score without improving the app.
- Local models ran on a laptop GPU (GTX 1650 Ti); cloud models are much faster.
