from enum import Enum

class ActivityType(str, Enum):
    ADD = 'Add'
    UPLOAD = 'Upload'
    PROCESS= 'Process'
    REVIEW = 'Review'
    MODIFY = 'Modify'
    DELETE = 'Delete'
    CANCEL = 'Cancel'
    STOP = 'Stop'
