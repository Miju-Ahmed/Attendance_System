from typing import Optional
from uuid import UUID, uuid4
from pydantic import BaseModel, Field

class EmotionStats(BaseModel):
    happy: float = 0.0
    sad: float = 0.0
    neutral: float = 0.0
