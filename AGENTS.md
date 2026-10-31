# DogDoor — Project Memory (for AI coding assistants)

A smart pet door for three toy poodles. The plan lives in `SPEC.md`; the **live state lives in
`STATUS.md`** (read it first). This file is the operating brief for any AI coding assistant.

## Operating rules

- Work the **next task in `STATUS.md`**, not ad hoc features.
- **Safety principle**: vision and RFID may *suggest* an open; a deterministic microcontroller
  *enforces* safety. The ML pipeline never drives the motor directly.
- **Generalizable-first**: no hard-coded animals — all classes derive from `config/roster.yaml`.
- Keep personal photos out of git (`data/raw/`, `runs/`, `models/` are gitignored).
- Reusable code in `src/`, docs in `docs/`, plan in `SPEC.md`, live state in `STATUS.md`.
- Run a syntax check + smoke test before finishing a slice; commit after each coherent slice.

## Harness

Planner (deepseek-v4-pro) writes specs and reviews; implementer (deepseek-v4.1-flash) writes code;
explore (read-only) researches; reviewer (gpt-6-luna, optional) gives second opinions on safety/ML.

Full setup and cost-effective delegation rules: **`docs/agent_orchestration.md`**. The enforceable
permission profiles live in **`.opencode/`** (`agents/general.md`, `explore.md`, `reviewer.md`, and
`opencode.jsonc`, which whitelists subagent task delegation). Start opencode from the project root so
the profiles load; restart after changing them.

## File ownership

- **Planner-owned** (implementers never edit): `SPEC.md`, `STATUS.md`, `AGENTS.md`, `README.md`,
  `docs/*`, `config/roster.yaml`, and git commits.
- **Implementer-owned**: `src/**`, `requirements.txt`, templates.

## Verification

```powershell
python -m compileall -q src
# smoke: all studio pages render
python -c "from fastapi.testclient import TestClient; from src.studio.app import app; c=TestClient(app); print([c.get(p).status_code for p in ['/','/datasets','/train','/predict','/experiments','/models']])"
# fast training smoke (tiny synthetic data only)
python -m src.vision.train --backbone mobilenet_v3_small --epochs 1 --img-size 32 --batch-size 4
```

## Cost control

Delegate read-only research to explore; implement with general; reserve reviewer for
safety/ML decisions. Do not re-read source files an explore subagent already summarised.
