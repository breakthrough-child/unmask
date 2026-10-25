# Lambda container image: OpenCV 5 (headless) + YuNet model baked in.
FROM public.ecr.aws/lambda/python:3.12

COPY requirements-lambda.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements-lambda.txt

COPY models/face_detection_yunet_2023mar.onnx /opt/models/face_detection_yunet_2023mar.onnx
ENV UNMASK_YUNET_MODEL=/opt/models/face_detection_yunet_2023mar.onnx

COPY unmask/ ${LAMBDA_TASK_ROOT}/unmask/
COPY backend/ ${LAMBDA_TASK_ROOT}/backend/

# Fail the build if OpenCV is not 5.x or the model checksum does not match.
RUN python -c "import cv2, unmask.vision.face as f; assert cv2.__version__.startswith('5'), cv2.__version__; assert f.verify_model(), 'YuNet checksum mismatch'; print('OpenCV', cv2.__version__)"

CMD ["backend.app.handler"]
