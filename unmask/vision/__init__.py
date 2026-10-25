"""OpenCV 5 vision signals for Unmask. Pure Python + OpenCV + NumPy, no AWS."""
from .analyze import CHALLENGES, analyze_clip  # noqa: F401
from .video import Clip, clip_from_frames, load_clip  # noqa: F401
