"""Face detection and five-point landmarks via OpenCV's bundled YuNet.

The model file must keep its .onnx extension. cv2.dnn dispatches on the
extension, and .bin is OpenVINO's weights format, so a YuNet model saved as
.bin sends OpenCV looking for an openvino backend it does not have and the
detector fails to construct. The flake names the fetched model accordingly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np

# YuNet's five landmarks, in the order the network emits them. "right" is the
# subject's right, which appears on the left of a non-mirrored frame.
LANDMARK_NAMES = ("right_eye", "left_eye", "nose", "right_mouth", "left_mouth")


@dataclass(frozen=True)
class Detection:
    score: float
    box: tuple[float, float, float, float]  # x, y, w, h in frame pixels
    landmarks: np.ndarray  # (5, 2) float32, frame pixels


def model_path() -> str:
    path = os.environ.get("FANCY_TRACKER_MODEL")
    if not path:
        raise RuntimeError(
            "FANCY_TRACKER_MODEL is not set. Run through `nix run` or inside "
            "`nix develop`, which both point it at the pinned YuNet model."
        )
    if not os.path.isfile(path):
        raise RuntimeError(f"FANCY_TRACKER_MODEL does not exist: {path}")
    if not path.endswith(".onnx"):
        raise RuntimeError(
            f"FANCY_TRACKER_MODEL must end in .onnx, got {path}. OpenCV picks its "
            "DNN importer from the extension and will not read this as ONNX."
        )
    return path


class FaceDetector:
    def __init__(self, score_threshold: float = 0.6, nms_threshold: float = 0.3, top_k: int = 500):
        self._detector = cv2.FaceDetectorYN.create(
            model_path(), "", (320, 320), score_threshold, nms_threshold, top_k
        )
        self._size: tuple[int, int] | None = None

    def detect(self, frame: np.ndarray) -> Detection | None:
        """Highest-scoring face in a BGR frame, or None if nothing was found."""
        h, w = frame.shape[:2]
        if self._size != (w, h):
            self._detector.setInputSize((w, h))
            self._size = (w, h)

        _retval, faces = self._detector.detect(frame)
        if faces is None or len(faces) == 0:
            return None

        # Rows are [x, y, w, h, 5 landmark xy pairs..., score], sorted by score.
        best = max(faces, key=lambda f: float(f[14]))
        landmarks = np.array(
            [[float(best[4 + 2 * k]), float(best[5 + 2 * k])] for k in range(5)],
            dtype=np.float32,
        )
        return Detection(
            score=float(best[14]),
            box=(float(best[0]), float(best[1]), float(best[2]), float(best[3])),
            landmarks=landmarks,
        )
