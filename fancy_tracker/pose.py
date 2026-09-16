"""Head pose from YuNet's five landmarks.

The feature vector deliberately mixes two kinds of cue. solvePnP yaw/pitch is
the interpretable one but gets noisy at steep angles with only five points;
the normalised nose offsets are cruder but degrade gently. Calibration
standardises both, so whichever is better behaved in a given seat dominates.

All four features are translation- and scale-invariant, so leaning in or
shifting the chair does not move them - only actually turning the head does.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Generic head, millimetres, in OpenCV camera convention: x right, y down,
# z forward (away from camera). The nose tip is the origin and protrudes
# towards the camera, so the other points sit at positive z.
MODEL_POINTS = np.array(
    [
        [-32.0, -40.0, 35.0],  # right eye
        [32.0, -40.0, 35.0],  # left eye
        [0.0, 0.0, 0.0],  # nose tip
        [-28.0, 38.0, 30.0],  # right mouth corner
        [28.0, 38.0, 30.0],  # left mouth corner
    ],
    dtype=np.float64,
)

FEATURE_NAMES = ("yaw", "pitch", "nose_dx", "nose_dy")
N_FEATURES = len(FEATURE_NAMES)


@dataclass(frozen=True)
class Pose:
    yaw: float  # degrees, positive when the head turns towards frame right
    pitch: float  # degrees, positive when the chin drops
    roll: float
    nose_dx: float  # nose offset from the eye midpoint, in interocular units
    nose_dy: float

    @property
    def features(self) -> np.ndarray:
        return np.array([self.yaw, self.pitch, self.nose_dx, self.nose_dy], dtype=np.float64)


def camera_matrix(width: int, height: int) -> np.ndarray:
    """Rough intrinsics. A real calibration would be better, but the classifier
    only needs the mapping to be consistent, not metrically true."""
    focal = float(width)
    return np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def estimate(landmarks: np.ndarray, width: int, height: int) -> Pose | None:
    """Pose from the (5, 2) landmark array, or None if solvePnP fails."""
    image_points = np.asarray(landmarks, dtype=np.float64)
    if image_points.shape != (5, 2):
        return None

    ok, rvec, _tvec = cv2.solvePnP(
        MODEL_POINTS,
        image_points,
        camera_matrix(width, height),
        np.zeros((4, 1)),
        flags=cv2.SOLVEPNP_SQPNP,
    )
    if not ok:
        return None

    rotation, _ = cv2.Rodrigues(rvec)
    pitch, yaw, roll = cv2.RQDecomp3x3(rotation)[0]

    right_eye, left_eye, nose = image_points[0], image_points[1], image_points[2]
    eye_mid = (right_eye + left_eye) / 2.0
    interocular = float(np.linalg.norm(left_eye - right_eye))
    if interocular < 1e-3:
        return None

    return Pose(
        yaw=float(yaw),
        pitch=float(pitch),
        roll=float(roll),
        nose_dx=float((nose[0] - eye_mid[0]) / interocular),
        nose_dy=float((nose[1] - eye_mid[1]) / interocular),
    )
