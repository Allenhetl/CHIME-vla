"""CHIME-VLA: Cross-modal Hierarchical Memory VLA.

Architecture v2.1 — see ``chat/architecture_v2.1.md``.  Component map:

* [C1]  perception.vlm_backbone.VLMBackbone        — SigLIP-ViT + LoRA
* [C2]  perception.fifo_buffer.WorkBuffer          — FIFO ring buffer (M_work)
* [C3]  heads.geo_write.GeoWriteHead               — geometric memory write
* [C4]  heads.sem_write.SemWriteHead               — semantic memory write
* [C5]  heads.espc.ESPC                            — Episodic Salience Predictor
* [C6]  memory.geo_grid.GeoGrid                    — multi-res voxel store (M_geo)
* [C7]  memory.sem_bank.SemBank                    — slot bank w/ slot_free (M_sem)
* [C8]  readout.read_interface.ReadInterface       — cross-attn + trilinear sample
* [C9]  action.action_expert.ActionExpert          — π0 flow-matching head + LoRA
* [C10] (offline, in Hindsight repo)               — HCS-H γ̂ generator
* [C11] heads.prh.PRH                              — Predictive Read Head (training)
* [C12] heads.csm.CSM                              — Counterfactual Slot Mask (training)
* [C13] (training-only)                            — λ_1 schedule (loss, not a Module)

This package is a stub-only skeleton at M0; algorithm bodies arrive at M1+.
"""

__version__ = "0.1.0"
