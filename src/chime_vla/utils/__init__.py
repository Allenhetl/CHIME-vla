"""Shared utilities — distributed, losses, seeding, letterbox, git_info,
plus CHIME-specific grad_flow_check / memory_reset.

Files copied verbatim from Hindsight (CODE_STRUCTURE.md §6):
    distributed.py · losses.py · seeding.py · letterbox.py · git_info.py
"""

from chime_vla.utils.distributed import all_gather_concat, get_rank, get_world_size
from chime_vla.utils.git_info import get_git_commit, save_run_provenance
from chime_vla.utils.letterbox import letterbox, letterbox_batch
from chime_vla.utils.losses import masked_mse
from chime_vla.utils.seeding import seed_all

__all__ = [
    "all_gather_concat",
    "get_rank",
    "get_world_size",
    "get_git_commit",
    "save_run_provenance",
    "letterbox",
    "letterbox_batch",
    "masked_mse",
    "seed_all",
]
