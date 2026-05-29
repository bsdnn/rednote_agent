"""LlamaIndex NodePostprocessor applying persona hard-filter + soft-boost.

Reads the current persona from `_context.current_persona` (ContextVar).
Pluggable into `RetrieverQueryEngine.node_postprocessors`.
"""
import logging
from typing import Optional

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle

from backend.models.persona import UserPersona
from ._context import current_persona
from .persona_rules import passes_skin_type, passes_budget

logger = logging.getLogger(__name__)


class PersonaNodePostprocessor(BaseNodePostprocessor):
    """LlamaIndex NodePostprocessor applying persona hard-filter + soft-boost.

    Reads the current persona from `_context.current_persona` (ContextVar).
    Pluggable into `RetrieverQueryEngine.node_postprocessors`.
    """
    per_match_boost: float = 0.05

    @classmethod
    def class_name(cls) -> str:
        return "PersonaNodePostprocessor"

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> list[NodeWithScore]:
        persona: UserPersona | None = current_persona.get()
        if persona is None:
            return nodes

        def _filter(skip_skin: bool) -> list[NodeWithScore]:
            out = []
            for nws in nodes:
                meta = nws.node.metadata
                if meta.get("doc_type") != "product":
                    out.append(nws)
                    continue
                if not skip_skin and not passes_skin_type(meta, persona.skin_type):
                    continue
                if not passes_budget(meta, persona.budget):
                    continue
                out.append(nws)
            return out

        primary = _filter(skip_skin=False)
        product_count = sum(
            1 for n in primary if n.node.metadata.get("doc_type") == "product"
        )
        if product_count == 0 and persona.skin_type:
            logger.warning(
                "persona filter emptied products, dropping skin_type=%s",
                persona.skin_type,
            )
            primary = _filter(skip_skin=True)

        boosted: list[tuple[NodeWithScore, float]] = []
        for nws in primary:
            meta = nws.node.metadata
            boost = 0.0
            if persona.age_group and persona.age_group in (meta.get("age_groups") or []):
                boost += self.per_match_boost
            overlap = set(persona.preferences or []) & set(meta.get("effects") or [])
            boost += self.per_match_boost * len(overlap)
            new_score = (nws.score or 0.0) + boost
            boosted.append((NodeWithScore(node=nws.node, score=new_score), boost))

        # Sort by boosted score, with boost magnitude as tiebreaker so that
        # persona-aligned nodes outrank tied non-boosted nodes.
        boosted.sort(key=lambda t: (t[0].score or 0.0, t[1]), reverse=True)
        return [t[0] for t in boosted]
