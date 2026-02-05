from importlib import import_module

from .logger import get_logger
from .tracker import EuclideanDistTracker, FaceTracker
from .make_url import MakeUrl

_LAZY_MODULES = {
    "ArcFace": "utils.ArcFace",
    "SCRFD": "utils.SCRFD",
    "ResEmoteNet": "utils.ResEmoteNet",
}

_HELPER_EXPORTS = {
    "estimate_norm",
    "norm_crop_image",
    "distance2bbox",
    "distance2kps",
    "compute_similarity",
    "draw_bbox",
    "draw_bbox_info",
    "resolve_local_media_path",
    "open_video_capture",
}

__all__ = [
    "get_logger",
    "ArcFace",
    "SCRFD",
    "ResEmoteNet",
    "EuclideanDistTracker",
    "FaceTracker",
    "MakeUrl",
    *sorted(_HELPER_EXPORTS),
]


def __getattr__(name):
    if name in _LAZY_MODULES:
        module = import_module(_LAZY_MODULES[name])
        attr = getattr(module, name)
        globals()[name] = attr
        return attr
    if name in _HELPER_EXPORTS:
        helpers = import_module("utils.helpers")
        attr = getattr(helpers, name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module 'utils' has no attribute '{name}'")