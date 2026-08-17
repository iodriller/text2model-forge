# Support matrix

Status meanings:

- **supported**: continuously tested or repeatedly reproduced on the stated target;
- **qualified**: one or more real evidence records exist, but the broader product path is incomplete;
- **experimental**: implemented with incomplete live evidence;
- **blocked**: must not be used for release output.

## Control plane

| Target | Status | Evidence / limitation |
|---|---|---|
| Windows, Python 3.12 | supported | Full deterministic suite and launcher tests |
| Ubuntu, Python 3.12 | supported | Full deterministic CI suite and demo |
| macOS, Python 3.12 | experimental | Bash and core code are portable; hosted macOS CI and live worker evidence are pending |
| Docker Studio + Ollama | experimental | Compose/configuration and container health are tested; GPU-heavy ComfyUI remains host-native |
| Remote or multi-user Studio | blocked | No authentication or authorization boundary; loopback-only by default |

## Generation stages

| Scope | Status | Evidence / limitation |
|---|---|---|
| D0–D10 orchestration | supported | Deterministic providers exercise contracts, gates, hashing, rollback, and invalidation |
| SDXL/Qwen concept generation | experimental | Live local runs exist; no published multi-prompt quality corpus |
| Hunyuan3D on 8 GB-class NVIDIA | qualified, restricted | One real concept-to-geometry run; community license is territory-restricted; full chain not qualified |
| TRELLIS.2 on RTX 5090/WSL2 | qualified | Single partial smoke record; official upstream requirements are higher and compatibility patches were needed |
| TripoSG / InstantMesh | qualified or discovered | Individual smoke records exist; quality and reproducibility remain incomplete |
| Blender cleanup/rig probes | qualified | Individual topology and deformation probes passed; identity, materials, motion, and engine import remain incomplete |
| Static-prop live golden path | experimental | Corpus and acceptance report are defined but not yet complete |
| Animated D0–D10 output | experimental | No complete real-hardware run |
| Globally cleared default model stack | blocked | No current stack passes both global license clearance and complete product qualification |

The machine-readable source of truth is `resources/workers/*.json` plus the
corresponding `resources/qualifications/*.json`; this document must never
upgrade a lifecycle beyond those records.
