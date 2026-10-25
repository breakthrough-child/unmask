#!/usr/bin/env bash
# Build the Lambda image inside AWS (CodeBuild -> ECR) and deploy the stack with it.
# No local Docker and no large image upload: only the build stack and the template go up.
#   scripts/remote_deploy.sh [stack] [region]      (commit + push first: CodeBuild clones GitHub)
set -euo pipefail
STACK="${1:-unmask}"
REGION="${2:-eu-north-1}"
BUILD_STACK="${STACK}-build"
export SAM_CLI_TELEMETRY=0
cd "$(dirname "$0")/.."

REF="$(git rev-parse HEAD)"
if ! git branch -r --contains "$REF" | grep -q origin/; then
  echo "Commit $REF is not on GitHub yet: run 'git push' first (CodeBuild clones the public repo)." >&2
  exit 1
fi

echo "== 1/4 image builder stack ($BUILD_STACK)"
aws cloudformation deploy --region "$REGION" --stack-name "$BUILD_STACK" --template-file build.yaml \
  --capabilities CAPABILITY_IAM --no-fail-on-empty-changeset
bout() { aws cloudformation describe-stacks --region "$REGION" --stack-name "$BUILD_STACK" \
          --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue" --output text; }
REPO_URI="$(bout ImageRepoUri)"; PROJECT="$(bout ProjectName)"

TAG="${REF:0:12}"
if aws ecr describe-images --region "$REGION" --repository-name "${REPO_URI#*/}" --image-ids imageTag="$TAG" >/dev/null 2>&1; then
  echo "== 2/4 image $REPO_URI:$TAG already built"
else
  echo "== 2/4 building $REF in CodeBuild ($PROJECT)"
  BUILD_ID="$(aws codebuild start-build --region "$REGION" --project-name "$PROJECT" \
    --environment-variables-override "name=GIT_REF,value=$REF,type=PLAINTEXT" --query build.id --output text)"
  while true; do
    read -r STATUS PHASE < <(aws codebuild batch-get-builds --region "$REGION" --ids "$BUILD_ID" \
      --query 'builds[0].[buildStatus,currentPhase]' --output text)
    echo "   $STATUS $PHASE"
    [ "$STATUS" = "IN_PROGRESS" ] || break
    sleep 20
  done
  if [ "$STATUS" != "SUCCEEDED" ]; then
    echo "CodeBuild $STATUS. Logs:" >&2
    aws logs tail "$(aws codebuild batch-get-builds --region "$REGION" --ids "$BUILD_ID" \
      --query 'builds[0].logs.groupName' --output text)" --region "$REGION" --since 1h | tail -40 >&2
    exit 1
  fi
fi
IMAGE_URI="$REPO_URI:$TAG"

echo "== 3/4 template with ImageUri=$IMAGE_URI"
mkdir -p .aws-sam
.venv/bin/python - "$IMAGE_URI" <<'PY'
import re, sys
s = open("template.yaml").read()
s = re.sub(r"\n    Metadata:\n      Dockerfile: Dockerfile\n      DockerContext: \.\n      DockerTag: [^\n]+", "", s)
s = s.replace("      PackageType: Image\n", f"      PackageType: Image\n      ImageUri: {sys.argv[1]}\n", 1)
open(".aws-sam/remote-template.yaml", "w").write(s)
PY

echo "== 4/4 sam deploy $STACK ($REGION)"
sam deploy --template-file .aws-sam/remote-template.yaml --stack-name "$STACK" --region "$REGION" \
  --resolve-s3 --capabilities CAPABILITY_IAM --no-fail-on-empty-changeset --no-confirm-changeset \
  --parameter-overrides "BedrockModelId=${BEDROCK_MODEL_ID:-eu.amazon.nova-lite-v1:0}"
aws cloudformation describe-stacks --region "$REGION" --stack-name "$STACK" --query "Stacks[0].Outputs" --output table
