import urllib.parse
from pathlib import Path

from environs import Env

from utils.logger import get_logger

env = Env()
env.read_env()

logger = get_logger(__name__)


class MakeUrl:
    def invoke(
        self,
        protocol: str | None = None,
        host: str | None = None,
        path: str | None = None,
    ) -> str:
        if path is None:
            raise ValueError("Path is required to generate URL")

        if env.bool("DEBUG", False):
            protocol = "http"
        else:
            protocol = "https"

        file_path = Path(path)
        try:
            mediafiles_index = file_path.parts.index("mediafiles")
            relative_path = file_path.parts[mediafiles_index:]
        except ValueError:
            logger.error("The path does not contain 'mediafiles'.")
            raise ValueError("The path does not contain 'mediafiles'.")

        if host is None:
            host = "localhost:8000"

        full_url = f"{protocol}://{host}/{Path(*relative_path).as_posix()}"
        logger.info(f"Url for the processed video path: {full_url}")
        return str(full_url)
