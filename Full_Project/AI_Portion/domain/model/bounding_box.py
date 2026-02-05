from pydantic import BaseModel

class BoundingBox(BaseModel):
    x: int
    y: int
    width: int
    height: int 
    track_id: int | None = None
    class_id: int | None = None
    confidence: float | None = None

    @property
    def center(self) -> tuple[int, int]:
        return int(self.x + self.width // 2), int(self.y + self.height // 2)

    @property
    def crop_rect(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.x + self.width, self.y + self.height