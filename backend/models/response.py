from pydantic import BaseModel, model_validator


class GenerateResponse(BaseModel):
    title: str
    body: str
    hashtags: list[str] = []
    emojis: list[str] = []

    @model_validator(mode="after")
    def normalize(self) -> "GenerateResponse":
        self.title = self.title.strip()
        self.body = self.body.strip()
        self.hashtags = [
            tag.strip() if tag.strip().startswith("#") else f"#{tag.strip()}"
            for tag in self.hashtags
            if tag.strip()
        ]
        self.emojis = [e.strip() for e in self.emojis if e.strip()]
        return self
