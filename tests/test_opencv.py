import cv2
import numpy as np

from unmask.vision.face import FaceDetector, model_path, verify_model


def test_opencv_major_version_is_5():
    assert cv2.__version__.startswith("5"), cv2.__version__


def test_required_apis_exist():
    for name in ("FaceDetectorYN", "solvePnP", "projectPoints", "Rodrigues", "calcOpticalFlowFarneback",
                 "createBackgroundSubtractorMOG2", "Laplacian", "Sobel", "VideoCapture"):
        assert hasattr(cv2, name), name


def test_yunet_model_checksum():
    assert verify_model(model_path())


def test_yunet_runs_on_blank_frame():
    det = FaceDetector()
    assert det.detect(np.zeros((480, 640, 3), np.uint8)) == []
