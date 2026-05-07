# CHIME-VLA Progress Log

## Current state

- **Milestone**: M0/M1/M2/M3 ALL PASS (MVP fallback path) → ready for M4
- **Active branch**: `m1/forward-impl` (推 GitHub, 18+ commits)
- **Updated**: 2026-05-08
- **Mode**: Autonomous orchestrator

## Progress arc (autonomous from milestone gate to M3 PASS)

| Step | Result |
|---|---|
| Setup Stage 0-4 | Repo + arch v2.1 + IMPLEMENTATION_PLAN + 13 component skeleton + orchestrator |
| M0 cache build | 379 LIBERO ep, 15.28 GB, 101k frames |
| M1 forward (9 comp) | 17.6M trainable params, full smoke train OK |
| M1 E1 cascade | untrained 0.173 → 200step 0.202 → norm 0.227 → 800step 0.177 (overfit) |
| M1 E1 verdict | HARD-FAIL → §0.7.4 MVP fallback applied (λ_1=0 永久, drop [C10][C12][C13]) |
| C11 PRH + C12 CSM | M2/M3 prep, 5.3M+ params, sem write head with slot_free |
| xfail cleanup | tests 12+10xpass+2xfail → **22 passed + 2 xfailed** strict |
| C5 self-supervised | L_predict — ψ trains via prediction error (not L_HCS) |
| **M2 PASS** | L_PRH 214× ↓, L_predict 8× ↓ |
| M3 grad flow fix | C3/C4 out-of-place memory updates, C3/C4 grad alive |
| M3 slot_free fix | argmax replaces threshold (was deadlocked at K_s=64) |
| **M3 PASS** | L_PRH per-k each 200×+ ↓, M_geo 2.6%, write head grad live |

## Most recent operations (last 8)

| ts | agent | action | result | commit |
|---|---|---|---|---|
| 2026-05-08 | impl × 9 | 9 forward components | green | 5fdd3f4 |
| 2026-05-08 | exp-runner | E1 cascade (4 runs) | HARD-FAIL → §0.7.4 fallback | 7be0f24 |
| 2026-05-08 | impl | C11 PRH + C12 CSM + L_PRH/L_CSM | green | 817c014 |
| 2026-05-08 | main | xfail cleanup | 22 passed + 2 xfail | e833be9 |
| 2026-05-08 | impl | C5 L_predict + analysis script | M2 PASS | 4aba7bc |
| 2026-05-08 | impl | C3/C4 grad flow + per-k L_PRH + slot_free argmax | **M3 PASS** | fbf019a |

## Blockers

- (none) — autonomous progression continues through M3

## Next action (M4)

**M4 deliverable per IMPLEMENTATION_PLAN §6** (LIBERO-only, MVP fallback adapted):
1. Full multi-loss training on LIBERO-Long (extend from 200 → 2000+ steps)
2. Held-out validation loss curves
3. SR evaluation on LIBERO held-out tasks
4. (Optional) OpenVLA + history baseline for delta comparison

**M4 challenges**:
- Long training run (~40-80 min for 1000-2000 steps on 2×4090)
- LIBERO simulator integration for true SR evaluation (not yet wired)
- Baseline runner (OpenVLA fine-tune or pulled checkpoint)

**Pragmatic M4 path** (autonomous):
- Phase 4a: longer training + val loss tracking (training is straightforward)
- Phase 4b: implement offline SR proxy (held-out action MSE as SR proxy until simulator wired)
- Phase 4c: defer full SR evaluation + OpenVLA baseline to "production stage" (when 6×A800 available)

**ETA**: 30-60 min for Phase 4a + 4b autonomous work

## Milestone gate status

### Setup (Stage 0-4)
- [x] All complete

### M0 — Repo skeleton + Hindsight 契约
- [x] cache 379 ep / 15.28 GB
- [x] 9 component imports OK + skeleton tests
- [x] git push to GitHub

### M1 — Forward implementation + smoke + E1
- [x] [C1-C9] forward implemented (17.6M trainable)
- [x] [C11] PRH + [C12] CSM (M2 prep)
- [x] train_step + 5-loss assembly
- [x] datamodule (with action normalization) + LightningModule
- [x] smoke training: 800 step OK
- [x] **E1 cascade — HARD-FAIL → MVP fallback applied** (architectural existential decision concluded)

### M2 — [C5] + [C11] independent (REDUCED per MVP fallback) ✓ PASS
- [x] [C5] L_predict self-supervised (ψ trains via h_t prediction MSE)
- [x] [C11] L_PRH 214× monotonic decrease verified
- [x] grad-flow CI strict pass

### M3 — joint training (REDUCED) ✓ PASS
- [x] L_main + L_PRH + L_predict joint training
- [x] C3/C4 write head grad flow alive (out-of-place memory updates)
- [x] M_geo occupancy 2.6% (target 1-10% ✓)
- [x] L_PRH per-k log (k=4 / k=16) — each 200×+ ↓
- [x] slot_free argmax fix (M_sem from 0% → 1.56%)

### M4 — full training + LIBERO SR baseline (NEXT)
- [ ] Phase 4a: longer training + val curves
- [ ] Phase 4b: offline SR proxy
- [ ] Phase 4c: defer to production stage (LIBERO sim + OpenVLA baseline)

### M6 — Ablation (3-loss subset, post-M4)

## GitHub
- 仓库: https://github.com/Allenhetl/CHIME-vla.git
- Branches: main + 4 stage + m1/forward-impl
- m1/forward-impl: 18 commits; latest: `fbf019a` M3 PASS

## Key empirical learnings (for paper writing later)

1. **Simple Jacobian saliency proxy doesn't recover sub_task boundaries** on LIBERO (peak IoU 0.227 at quantile 0.3 vs target 0.4) — confirms architecture's §I.4 #1 risk; full HCS-H with RUDDER + grad-cam likely required for hindsight signal
2. **Action normalization helps slightly** (+0.025 IoU) but doesn't fix fundamental signal weakness
3. **800-step over-training regresses IoU** — confirms our saliency proxy isn't the right one for E1
4. **C5 self-supervised L_predict works alone** — 8× MSE drop in 200 steps; ψ can train without L_HCS
5. **slot_free + threshold > 0.1 was deadlocked** at K_s=64 (softmax too flat); argmax breaks the deadlock
6. **GeoGrid sparse-write invariant holds** — 2.6% voxel occupancy across 200 steps (architecture target 1-10%)
7. **L_PRH 200×+ reduction in 151 steps** — predictive read learning works fast
