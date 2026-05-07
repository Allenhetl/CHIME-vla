# CHIME-VLA Progress Log

## Current state

- **Milestone**: M1 完成 → **MVP fallback applied** → 进入 M2 (reduced loss set)
- **Active branch**: `m1/forward-impl` (推 GitHub)
- **Updated**: 2026-05-08
- **Mode**: Autonomous orchestrator
- **Critical decision**: M1 E1 cascade HARD-FAIL → architecture §0.7.4 + §I.4 #1 fallback applied

## Recent operations (last 8)

| ts | agent | action | result | commit |
|---|---|---|---|---|
| 2026-05-07 | implementer × 9 | M1 forward 9 components (C1-C9) | green; full chain ✓ | 5fdd3f4 |
| 2026-05-08 | experiment-runner | E1 untrained baseline (5 ep) | IoU @ 0.3 = 0.173 ≈ random | d27ad01 |
| 2026-05-08 | experiment-runner | E1 200 step trained (no norm) | IoU @ 0.3 = 0.202 (+0.029) | cdcbeb8 |
| 2026-05-08 | implementer | A2 action normalization | datamodule + stats.json | 19d0175 |
| 2026-05-08 | experiment-runner | E1 200 step + norm (A2) | IoU @ 0.3 = 0.227 (peak) | (output/reports) |
| 2026-05-08 | experiment-runner | E1 800 step + norm (A1+) | IoU @ 0.3 = **0.177** REGRESSED (overfit) | (output/reports) |
| 2026-05-08 | implementer × 2 | C11 PRH + C12 CSM + L_PRH/L_CSM wiring | M2 prep ready, 22 passed + 2 xfail | 817c014 |
| 2026-05-08 | main | xfail cleanup (10 XPASS → strict pass) | tests 22+2 cleanly | e833be9 |

## E1 Cascade — Final Decision

**Architecture §I.4 #1 (30-50% probability) fallback path activated:**

| Run | IoU(main) | IoU @0.3 | IoU @0.5 | IoU @0.7 | random |
|---|---|---|---|---|---|
| untrained | 0.146 | 0.173 | 0.258 | 0.307 | 0.169 |
| 200 step | 0.188 | 0.202 | 0.273 | 0.312 | 0.169 |
| 200 step + norm | **0.197** | **0.227** | 0.280 | 0.317 | 0.169 |
| 800 step + norm | 0.149 | 0.177 | 0.241 | 0.309 | 0.169 |

**Verdict**: Cascade peak IoU @ 0.3 = 0.227 < 0.3 SOFT-PASS bar < 0.4 PASS bar = **HARD-FAIL**.

**Root cause**: Simple Jacobian saliency `∂a_{t+Δ}/∂o_t` magnitude does NOT align with sub_task_id boundaries on LIBERO. Architecture's full [C10] HCS-H needs RUDDER + grad-cam decomposition to recover task structure. Empirically: 800 step training **regressed** IoU below 200 step (overfit).

**Architecture-prescribed fallback (§0.7.4 + §I.4 #1)**:
- λ_1 = 0 permanently → drop L_HCS
- drop [C10] HCS-H, [C12] CSM, [C13] reverse Jacobian
- keep: L_main + L_PRH + L_aux (3-loss MVP)
- Publishable claim retreats to "Event-segmentation prediction error 双通道 delta-rule + MERLIN-style predictive read" (architecture line 1979)

## Blockers

- (none) — autonomous progression resumed under MVP fallback

## Next action

**M2 stage** with reduced loss set:
1. Implement C5 self-supervised prediction loss (ψ predicts h_t from M_work, MSE)
2. Train M2 with `+train=m2_mvp_fallback`
3. Verify L_PRH @ k=4,16 monotonic decrease
4. Verify [C11] PRH gradient signal alive
5. Skip M2 [C5] IoU > 0.5 verification (since L_HCS off, γ_sem self-supervised only)

**ETA**: 1-2 sessions of dispatched agent work + ~30 min training

## Milestone gate status

### Setup (Stage 0-4)
- [x] All complete

### M0 — Repo skeleton + Hindsight 契约
- [x] Complete (cache 379 ep + 9 component imports + skeleton tests)

### M1 — Forward implementation + smoke + E1
- [x] [C1-C9] forward all implemented (17.6M trainable params)
- [x] [C11] PRH + [C12] CSM implemented (M2 prep)
- [x] train_step + 5 losses (L_HCS sentinel, L_PRH/L_CSM real, L_main + L_aux real)
- [x] datamodule + LightningModule + scripts/10_train.py
- [x] smoke training: 800 step on LIBERO + checkpoint OK
- [x] **E1 cascade — HARD-FAIL → MVP fallback applied**
- [x] action normalization (datamodule loads stats.json)

### M2 — [C5] + [C11] independent (REDUCED per MVP fallback)
- [ ] C5 self-supervised prediction loss (no L_HCS, ψ trains on prediction error only)
- [ ] C11 L_PRH @ k=4,16 monotonic decrease verification
- [ ] config: `m2_mvp_fallback.yaml` already created

### M3 — joint training (REDUCED)
- [ ] L_main + L_PRH joint, no L_HCS / L_CSM
- [ ] write head gradient flow verification

### M4 — full reduced 3-loss training + LIBERO SR baseline
- [ ] L_main + L_PRH + L_aux on LIBERO-Long
- [ ] vs OpenVLA + 8-frame history baseline, SR Δ ≥ 10%

### M6 — Ablation (reduced set)
- [ ] subset of 10 ablations applicable to 3-loss config

## GitHub
- 仓库: https://github.com/Allenhetl/CHIME-vla.git
- 分支: main + 4 stage + m1/forward-impl (12+ commits, all pushed)
- 最新: m1/forward-impl with E1 cascade conclusion + xfail cleanup
