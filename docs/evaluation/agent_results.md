# Repair agent: fault-injection benchmark

A working case is generated, one realistic fault is injected (most are bugs this project actually had), and the agent must make the case run again ([`evals/faults.py`](../../evals/faults.py)). ✅ repaired (by which fixer), ❌ not repaired.

| Fault | rules only | ollama · qwen2.5:7b (rules + LLM) | gemini · gemini-3.1-flash-lite (rules + LLM) |
|---|:-:|:-:|:-:|
| pimpleFoam without kFinal/epsilonFinal solver entries (original bug) | ✅ rules | ✅ rules | ✅ rules |
| interFoam without the pcorr solver (original bug) | ✅ rules | ✅ rules | ✅ rules |
| interFoam scheme missing in v1912 (original bug: OpenFOAM.org syntax) | ✅ rules | ✅ rules | ✅ rules |
| convection scheme for k deleted | ✅ rules | ✅ rules | ✅ rules |
| buoyantSimpleFoam with SIMPLEC and weak relaxation: negative temperature (original bug) | ✅ rules | ✅ rules | ✅ rules |
| Mach 0.5 rhoSimpleFoam, aggressive relaxation and no pressure limits | ✅ rules | ✅ rules | ✅ rules |
| field patch type does not match the mesh patch (original STL bug) | ❌ | ✅ LLM | ✅ LLM |
| fixedValue outlet pressure without a value | ❌ | ❌ | ✅ LLM |
| unclosed brace in fvSchemes (hand-editing mistake) | ❌ | ❌ | ✅ LLM |
| typo in a linear-solver name | ❌ | ✅ LLM | ✅ LLM |
| **Repaired** | **6/10** | **8/10** | **10/10** |
