from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class User(BaseModel):
    user_id: UUID | None = None
    user_name: str | None = None
    user_email: str | None = None
    user_role: str | None = None    