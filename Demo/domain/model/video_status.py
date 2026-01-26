from enum import Enum

class VideoStatus(str, Enum):
    PROCESSING = "Processing"
    PROCESSED = "Processed"
    FAILED = "Failed"
    UNPROCESSED = "Unprocessed"