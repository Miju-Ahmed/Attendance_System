from pydantic import BaseModel
from .user import User

class UserData(BaseModel):
    user_created: User | None = None
    user_modified: User | None = None