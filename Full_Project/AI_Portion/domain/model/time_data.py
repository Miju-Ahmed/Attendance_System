from pydantic import BaseModel
from datetime import datetime , timedelta , date , time

class TimeData(BaseModel):
    created_at: datetime | None = None
    modified_at: datetime | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    presence_time: int | None = None
    absence_time: int | None = None
    duration: timedelta | None = None
    date_of_activity: date | None = None

    office_start_time: time | None = None
    office_end_time: time | None = None
    
    
    
    
    
    
    