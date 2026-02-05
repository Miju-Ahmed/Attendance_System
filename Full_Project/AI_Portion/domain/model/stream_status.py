from enum import Enum

class StreamStatus(str, Enum):
    UPLOADING = 'Uploading'
    PROCESSING = 'Processing'
    PROCESSED = 'Processed'
    FAILED = 'Failed'
    STOPPED = 'Stopped'
    