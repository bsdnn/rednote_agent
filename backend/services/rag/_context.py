"""ContextVar for persona injection into the LlamaIndex pipeline.

Lives in a separate module to avoid circular imports between
`service.py` and `persona_postprocessor.py`.

`service.query_product_database` sets the contextvar before calling
`engine.aquery(query)`. `PersonaNodePostprocessor` reads it during
node post-processing.
"""
from contextvars import ContextVar
from typing import Optional

from backend.models.persona import UserPersona

current_persona: ContextVar[Optional[UserPersona]] = ContextVar(
    "rag_current_persona",
    default=None,
)
