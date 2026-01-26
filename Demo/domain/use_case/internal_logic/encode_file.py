"""Use case for encoding processed videos.

This module provides a simple wrapper around ``ffmpeg`` so that processed
videos can be transcoded into a frontend-friendly format immediately after
the analytics pipeline finishes writing to ``processed_uri``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple

from utils import get_logger


logger = get_logger(__name__)


class EncodeFile:
    """Encode a video file using ``ffmpeg``.

    The encoder overwrites the source file with an H.264/AAC combination so it
    can be streamed by browsers.
    """

    def invoke(self, file_path: str | Path) -> bool:
        """Encode the provided file path.

        Args:
            file_path: Absolute or relative path to the video that needs to be
                transcoded.

        Returns:
            ``True`` when encoding succeeds, ``False`` otherwise.
        """

        resolved_path = Path(file_path).expanduser().resolve()
        logger.info("Encoding processed video at %s", resolved_path)

        temp_output = resolved_path.with_name(
            f"{resolved_path.stem}_encoded{resolved_path.suffix}"
        )

        if temp_output.exists():
            temp_output.unlink(missing_ok=True)

        color_flags, frame_rate = self._probe_video_metadata(resolved_path)

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(resolved_path),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-strict",
            "experimental",
            "-b:a",
            "192k",
        ]

        # Preserve the input color profile so that the encoded output looks the
        # same as the original capture.  Without these flags ffmpeg falls back
        # to defaults (BT.709 limited range) which can noticeably shift colors.
        for option, value in color_flags.items():
            cmd.extend([option, value])

        if frame_rate:
            cmd.extend(["-r", frame_rate])

        cmd.append(str(temp_output))

        try:
            logger.info("Running ffmpeg to encode processed video")
            subprocess.run(cmd, check=True)
            temp_output.replace(resolved_path)
            return True
        except subprocess.CalledProcessError as exc:  # pragma: no cover - defensive
            logger.error("FFmpeg failed to encode %s: %s", resolved_path, exc)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Unexpected error while encoding %s: %s", resolved_path, exc)
        finally:
            if temp_output.exists():
                temp_output.unlink(missing_ok=True)

        return False

    def _probe_video_metadata(self, video_path: Path) -> Tuple[Dict[str, str], Optional[str]]:
        """Collect color metadata and frame rate from the source video via ``ffprobe``.

        ffmpeg exposes color control flags via ``-color_primaries``,
        ``-color_trc``, ``-colorspace`` and ``-color_range``.  ``ffprobe``
        outputs slightly different names so we map them to the flag names used
        during encoding. We also capture ``avg_frame_rate`` so the encoder can
        preserve the original FPS instead of falling back to the default 25fps
        when the writer omitted an explicit rate. When probing fails we simply
        return defaults so encoding continues with ffmpeg defaults instead of
        aborting.
        """

        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=color_space,color_transfer,color_primaries,color_range,avg_frame_rate",
            "-of",
            "default=noprint_wrappers=1",
            str(video_path),
        ]

        mapping = {
            "color_primaries": "-color_primaries",
            "color_transfer": "-color_trc",
            "color_space": "-colorspace",
            "color_range": "-color_range",
        }

        try:
            logger.debug("Probing source color metadata with ffprobe")
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            logger.warning(
                "Unable to probe color metadata for %s (ffprobe error): %s",
                video_path,
                exc,
            )
            return {}, None

        color_values: Dict[str, str] = {}
        frame_rate: Optional[str] = None

        for line in result.stdout.strip().splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not value:
                continue

            if key == "avg_frame_rate":
                if value not in {"0/0", "N/A"}:
                    frame_rate = value
                continue

            option = mapping.get(key)
            if option:
                color_values[option] = value

        return color_values, frame_rate
