from __future__ import annotations
from enum import Enum
from pydantic import BaseModel, Field
from .persona import UserPersona


class Tone(str, Enum):
    lively  = "活泼甜美"
    pro     = "专业种草"
    healing = "温柔治愈"
    humorous = "幽默搞笑"
    elite   = "精英范儿"


class GenerateRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=500)
    tone: Tone = Tone.lively
    max_iterations: int = Field(default=5, ge=1, le=8)
    persona: UserPersona | None = None
    user_id: str | None = Field(default=None, max_length=64)


class RefineRequest(BaseModel):
    previous_result: dict
    refinement_instruction: str = Field(..., min_length=1, max_length=300)
    conversation_history: list[dict] = []
    persona: UserPersona | None = None
    tone: Tone = Tone.lively
