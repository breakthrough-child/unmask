PY ?= .venv/bin/python
STACK ?= unmask
REGION ?= us-east-1
PORT ?= 8000

.PHONY: setup test eval eval-synth samples serve build deploy deploy-frontend smoke check-bedrock diagrams clean

setup:            ## create venv, install pinned deps, fetch + verify YuNet model
	python3.12 -m venv .venv
	$(PY) -m pip install -r requirements-dev.txt
	@test -f models/face_detection_yunet_2023mar.onnx || curl -L -o models/face_detection_yunet_2023mar.onnx \
	  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
	$(PY) -c "import cv2, unmask.vision.face as f; assert cv2.__version__.startswith('5'); assert f.verify_model(); print('OpenCV', cv2.__version__, 'YuNet OK')"

test:             ## unit + pipeline tests
	$(PY) -m pytest

eval:             ## evaluate on data/ (your recordings)
	$(PY) eval/run_eval.py --data data --export-trace

eval-synth:       ## evaluate on generated synthetic fixtures (pipeline check)
	$(PY) eval/run_eval.py --synthetic --out eval/out/synthetic-results

samples:          ## copy sample clips into frontend/samples
	$(PY) scripts/make_samples.py

serve:            ## run frontend + API locally (no AWS needed)
	$(PY) -m backend.local_server --port $(PORT)

build:            ## build the Lambda container image (needs Docker + SAM CLI)
	sam build

deploy: build     ## deploy the stack (first time: sam deploy --guided)
	sam deploy --stack-name $(STACK) --region $(REGION) --resolve-image-repos --resolve-s3 \
	  --capabilities CAPABILITY_IAM --no-fail-on-empty-changeset

deploy-frontend:  ## upload frontend to S3 and invalidate CloudFront
	./scripts/deploy_frontend.sh $(STACK) $(REGION)

smoke:            ## hit the deployed API with a sample clip
	$(PY) scripts/smoke_test.py --stack $(STACK) --region $(REGION)

check-bedrock:    ## list Claude models / inference profiles usable in REGION
	$(PY) scripts/check_bedrock.py --region $(REGION)

diagrams:         ## render Mermaid diagrams to SVG/PNG (needs Chrome + internet)
	$(PY) scripts/render_diagrams.py

clean:
	rm -rf .aws-sam eval/out .pytest_cache .local-store
