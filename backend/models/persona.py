from typing import Literal
from pydantic import BaseModel


class UserPersona(BaseModel):
    skin_type: Literal["oily", "dry", "combination", "sensitive", "normal"] = "combination"
    age_group: Literal["18-24", "25-30", "31-40", "41+"] = "25-30"
    preferences: list[str] = []
    budget: Literal["budget", "mid-range", "luxury"] = "mid-range"
