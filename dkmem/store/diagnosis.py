"""Loss-point diagnosis hook for the store: the A-D harness's own ``diagnose``, per comparison.

``pair_diagnoser(lexicon, config_id)`` returns the ``diagnoser`` argument of
``dkmem.store.consolidate.consolidate``. It calls ``dkmem.pipeline.trace.diagnose`` with the
same arguments the harness uses (candidate = the earlier utterance's entry, new entry = the
later one), so a comparison's ``diagnosis`` equals the frozen trace row's. Like there, it is
lexicon-relative and says nothing about gold labels.
"""

from __future__ import annotations

from typing import Any, Callable

from dkmem.config import get_pipeline_config
from dkmem.memory.lexicon import Lexicon
from dkmem.pipeline.runner import _HOST_LOSS_STAGE, _STORAGE_LOSS_STAGE
from dkmem.pipeline.trace import diagnose
from dkmem.store.entry import StoredEntry

__all__ = ["pair_diagnoser"]


def pair_diagnoser(lexicon: Lexicon, config_id: str) -> Callable[[StoredEntry, StoredEntry, str], dict[str, Any]]:
    get_pipeline_config(config_id)  # validates the id
    storage_stage = _STORAGE_LOSS_STAGE[config_id] or "storage"
    host_stage = _HOST_LOSS_STAGE[config_id]

    def diagnoser(candidate: StoredEntry, new: StoredEntry, host_proposed: str) -> dict[str, Any]:
        return diagnose(
            lexicon=lexicon, lang=new.language,
            utterance_a=candidate.surface, utterance_b=new.surface,
            stored_text_a=candidate.stored_text, stored_text_b=new.stored_text,
            host_proposed=host_proposed, storage_loss_stage=storage_stage, host_loss_stage=host_stage,
        )

    return diagnoser
