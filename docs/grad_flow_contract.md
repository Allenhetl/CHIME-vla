# Grad-flow Contract (CI gate)

This file is the canonical specification of stop-gradient (sg) boundaries in CHIME-VLA. It is a contract between architecture and code: every sg listed here is enforced by a unit test in `tests/test_grad_flow.py`. A red CI run on this file BLOCKS all implementation PRs.

Source of truth: `chat/architecture_v2.1.md` §B (anchor `<!-- SG-MATRIX-CANONICAL -->`) and §B.1.

## SG-1..SG-7 (canonical, 7 entries)

| SG | Boundary | Direction | Test |
|---|---|---|---|
| **SG-1** | `[C3].write_(h_t, sg(γ_geo), M_geo)` and `[C4].write_(h_t, sg(γ_sem), M_sem)` | L_main must NOT flow back to ψ via γ | `test_sg_1_gamma_to_psi` |
| **SG-2** | `[C8] query projection` on L_PRH path | L_PRH must NOT flow to [C1] via query → h_t | `test_sg_2_prh_query_to_perception` |
| **SG-3** | `γ̂` produced by [C10] before being consumed by L_HCS | γ̂ is a target, not learnable | `test_sg_3_gammahat_target` |
| **SG-4** | `[C12]` calls `frozen_action_expert` (no grad pathway) | L_CSM must NOT train [C9] | `test_sg_4_csm_through_frozen_action` |
| **SG-5** | M_work content seen by ψ on L_HCS path | L_HCS must NOT flow to [C1] via h_{t-K_w..t-1} in M_work | `test_sg_5_mwork_to_perception_via_psi` |
| **SG-6** | [C5] geo_proj / sem_proj | Trainable ONLY by L_HCS, NOT L_main | `test_sg_6_proj_only_lhcs` |
| **SG-7** | [C8] cross-attn read end | NOT sg-able (structural); monitor `H(attn) > entropy_floor` | `test_sg_7_attention_entropy_floor` (runtime metric) |

## Test contract

Each `test_sg_<n>_*` follows this pattern (pseudo):

```python
def test_sg_1_gamma_to_psi():
    model = build_chime_full(cfg)
    # zero out all gradients
    model.zero_grad()
    # forward + selective loss
    out = forward_one_step(model, batch)
    L = compute_only_L_main(out)
    L.backward()
    # assertion: ψ params must have grad=None or zeros
    for name, p in model.heads.espc.psi.named_parameters():
        assert p.grad is None or p.grad.abs().max() < 1e-9, f"SG-1 violated: {name}"
    for name, p in model.heads.espc.geo_proj.named_parameters():
        assert p.grad is None or p.grad.abs().max() < 1e-9, f"SG-1/SG-6 violated: {name}"
    for name, p in model.heads.espc.sem_proj.named_parameters():
        assert p.grad is None or p.grad.abs().max() < 1e-9, f"SG-1/SG-6 violated: {name}"
```

## Why this is a CI gate (not a milestone gate)

A single sg violation silently degrades training behavior — L_main can leak into ψ, L_HCS can leak into [C1] LoRA — and these violations are invisible in loss curves until late in training. Verifying via gradient-flow assertion is cheap (~seconds per test) and catches refactor regressions immediately.

## Reference

- `chat/architecture_v2.1.md` §B `<!-- SG-MATRIX-CANONICAL -->` (canonical 7-row table)
- `chat/architecture_v2.1.md` §B.1 (test contract)
- `chat/architecture_v2.1.md` §B.2 (forward order pseudocode)
- `chat/chime_vla_proposal.md` §5.4 line 229-242 (original 7-row source)
