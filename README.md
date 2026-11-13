# CFD Case Generator

A Streamlit web application that translates natural-language CFD problem descriptions into complete, ready-to-run OpenFOAM case folders.  Supports two LLM backends: **Claude API** (cloud) and **Ollama** (local, free, no API key required).

![Screenshot placeholder](docs/screenshot.png)

---

## Supported Solvers

| Solver | Use Case | Flow Type |
|---|---|---|
| `simpleFoam` | Steady-state incompressible flow | Laminar or turbulent, single-phase |
| `pimpleFoam` | Transient incompressible flow | Unsteady, single-phase (PIMPLE algorithm) |
| `buoyantSimpleFoam` | Steady buoyancy-driven flow | Natural convection, heat transfer |
| `interFoam` | Two-phase immiscible flow | Free-surface, VOF method |

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2a. Claude API backend (optional — cloud)

Get an API key at [console.anthropic.com](https://console.anthropic.com) and either:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."   # pre-set in environment
```

…or paste it directly into the sidebar when the app is running.

### 2b. Ollama backend (optional — local, free)

#### Ollama setup

1. Install Ollama from https://ollama.com
2. Pull a model:
   ```bash
   ollama pull mistral
   ```
3. Start the server:
   ```bash
   ollama serve
   ```
4. Start the app normally and select **Ollama** in the sidebar.

Other models that work well: `llama3`, `qwen2.5`, `phi4`.

### 3. Run the app

```bash
streamlit run app/main.py
```

The app will open at `http://localhost:8501`.

---

## Example Input Descriptions

**simpleFoam — external aerodynamics:**
```
Steady-state airflow around a 2D cylinder at Re=200 with a freestream velocity
of 1 m/s. The fluid is air at standard conditions. I want to use a channel-type
domain, 20 diameters long and 10 diameters wide.
```

**pimpleFoam — transient vortex shedding:**
```
Unsteady incompressible flow past a bluff body at Re=1000. Freestream is water
at 0.5 m/s. I need to capture vortex shedding so I need a transient solver.
Domain is 30 chord lengths long, 15 wide.
```

**buoyantSimpleFoam — natural convection:**
```
Natural convection in a square cavity with hot left wall at 350 K and cold
right wall at 300 K. The fluid is air. Ra=1e6. Steady state solution needed.
```

**interFoam — dam break / free surface:**
```
Dam-break simulation with water column 0.3 m high and 0.1 m wide releasing
into a dry basin 1.0 m long. Gravity-driven two-phase (water/air) flow.
```

---

## Project Structure

```
cfd-case-generator/
├── app/
│   ├── main.py              # Streamlit entry point
│   ├── llm.py               # Claude + Ollama backends, unified router
│   ├── utils.py             # Zip helpers, file writing
│   └── generators/
│       ├── base.py          # Abstract base class
│       ├── simple_foam.py
│       ├── pimple_foam.py
│       ├── buoyant_simple_foam.py
│       └── inter_foam.py
├── templates/               # Reference OpenFOAM files (Jinja2 templates)
│   ├── simpleFoam/
│   ├── pimpleFoam/
│   ├── buoyantSimpleFoam/
│   └── interFoam/
├── tests/
│   └── test_generators.py
├── requirements.txt
└── README.md
```

---

## Known Limitations

- **Mesh generation:** The generated `blockMeshDict` is a simple parametric box/channel. Complex geometries (e.g., cylinders, airfoils) require additional meshing steps with `snappyHexMesh` or external tools.
- **Turbulence models:** Only the `kEpsilon` and `kOmegaSST` RAS models are supported in the templates. LES configurations are not generated.
- **2D cases:** The tool generates 3D cases with one cell in the span-wise direction. True 2D wedge geometries are not automated.
- **Solver version:** Templates target OpenFOAM v1912+ (ESI/openfoam.com) using `transportProperties` and `thermophysicalProperties`. OpenFOAM.org v10+ (which uses `physicalProperties` instead) is not directly supported.
- **Boundary conditions:** Complex inlet profiles (e.g., turbulent inflow, mapped BCs) are not generated — only uniform fixedValue / zeroGradient / noSlip.
- **Radiation:** `buoyantSimpleFoam` radiation is set to `none` by default; activate manually if needed.
- **Claude API key:** Required only for the Claude backend.  The Ollama backend works without any API key.
- **Ollama quality:** Local models (especially smaller ones) may produce less precise JSON than Claude.  If parsing fails the app retries up to 3 times automatically.  Larger models (≥7B parameters) give better results.
