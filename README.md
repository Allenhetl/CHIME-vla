# CHIME-VLA

Cross-modal Hierarchical Memory for Vision-Language-Action — an online VLA
system with a 13-component memory architecture (perception → ESPC saliency →
geometric / semantic memory → predictive read → flow-matching action expert,
with offline `Hindsight` saliency labels supervising the online ψ head).

The architecture spec is in `chat/architecture_v2.1.md`.  This repo
implements that spec across milestones M0 → M6 (see `IMPLEMENTATION_PLAN.md`).

## Quick start

```bash
# Editable install (PyTorch 2.1-2.5 + Lightning + Hydra; pin in pyproject)
pip install -e .[dev]

# Sanity check the package loads
python -c "from chime_vla.config import ChimeConfig; print(ChimeConfig().c1.backbone)"

# Run the test suite (M0: most contract tests xfail by design)
pytest -v

# Hydra config preview
python -c "import hydra; from hydra import compose, initialize; \
  initialize(config_path='configs', version_base=None); \
  print(compose(config_name='default'))"

# (Once data is staged) build the LIBERO cache
python scripts/00_build_libero_cache.py --help
```

## Documents

The single-source-of-truth set, in reading order:

| Doc | Purpose |
|---|---|
| `chat/architecture_v2.1.md` | Canonical architecture (components C1–C12, SG matrix, forward order) |
| `PLAN.md` | Project dashboard — milestone status at a glance |
| `IMPLEMENTATION_PLAN.md` | Per-milestone deliverables (M0 skeleton → M6 ablations) |
| `CODE_STRUCTURE.md` | Directory layout + Hydra config schema + interface signatures |
| `CODE_STANDARDS.md` | Engineering rules (sg matrix, dtype, loss reduction, λ_1 schedule) |
| `docs/grad_flow_contract.md` | SG-1..SG-7 unit-test specifications (CI gate) |
| `docs/hindsight_contract.md` | `gamma_hat.pt` file protocol between Hindsight and CHIME-VLA |
| `docs/data_schema.md` | LIBERO h5 → per-episode `.pt` cache schema |
| `PROGRESS.md` | Live engineering log |

## Repo map

```
CHIME-VLA/
├── src/chime_vla/        # Package — 13 component modules + training/
├── configs/              # Hydra YAML tree (default → model/train/data/experiment)
├── tests/                # CI gates: grad-flow, slot-lifecycle, forward-shapes, …
├── scripts/              # CLI entry points (data cache, train, eval, ablation)
├── docs/                 # Authoritative contracts (grad flow, hindsight, data)
├── chat/                 # Design docs (read-only)
└── Hindsight/            # Sibling repo: offline γ̂ labeler, file-protocol coupled
```

## Development workflow

* **Branch per milestone.**  `m0/skeleton` → `m1/e1-judgment` → `m2/espc-prh`
  → ... ; merge only after the milestone gate passes.  See `CODE_STANDARDS.md`
  §3 for the prefix table (`feat(C5):`, `fix(SG-N):`, `data:`, `exp(M2):`).
* **CI gates.**  `pytest tests/test_grad_flow.py` and
  `tests/test_slot_lifecycle.py` are blocking from M2 onward (`xfail` allowed
  in M0/M1).  `tests/test_forward_shapes.py` is strict from day one.
* **Hindsight coupling.**  CHIME-VLA does **not** import from `Hindsight/` —
  it consumes `output/saliency/gamma_hat/` files via
  `chime_vla.hindsight.consumer.HindsightConsumer` (see
  `docs/hindsight_contract.md`).
* **Provenance.**  Every training run writes
  `${output_root}/_hydra_runs/<exp>/<ts>/{config_resolved.yaml, git_commit.txt,
  requirements_freeze.txt}` (inherited from Hindsight `CODE_STANDARDS.md` §0.2).

## Self-driving execution (Stage 4)

Once M0 lands, this repo is driven by the `/loop` harness — Claude Code
(Opus) consumes `IMPLEMENTATION_PLAN.md`, picks the next sub-task, and
commits incrementally per milestone branch.  At M0 the loop is **not**
needed; manual command-line invocation is fine.  Stage-4 setup (cron / dyn
pacing / heartbeat hook) lands once M0 → M1 transition is clean.

## Hardware / data assumptions

* Dev box: 1 × H100 (80 GB) is comfortable for chime_full at bs=24; chime_mvp
  fits a single 40 GB card.
* LIBERO-Long raw h5 lives at `/home/sqmluser/data/memory_vla/libero_long/`
  (paths editable in `configs/data/libero_long.yaml`).  Cache is built once
  per repo, reused across milestones.

## License

Internal research code.  Do not redistribute.
