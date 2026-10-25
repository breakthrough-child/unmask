#!/usr/bin/env bash
# Upload frontend/ to the stack's S3 bucket, write config.js with the API URL, invalidate CloudFront.
set -euo pipefail
STACK="${1:-unmask}"
REGION="${2:-eu-north-1}"
out() { aws cloudformation describe-stacks --stack-name "$STACK" --region "$REGION" \
          --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue" --output text; }
API=$(out ApiUrl); BUCKET=$(out FrontendBucketName); DIST=$(out DistributionId); URL=$(out FrontendUrl)
REPO="${REPO_URL:-}"
echo "window.UNMASK_CONFIG = {apiUrl: \"$API\", mode: \"aws\", repoUrl: \"$REPO\"};" > frontend/config.js
aws s3 sync frontend/ "s3://$BUCKET/" --region "$REGION" --delete \
  --exclude "config.js" --cache-control "public,max-age=300"
aws s3 cp frontend/config.js "s3://$BUCKET/config.js" --region "$REGION" --cache-control "no-store"
aws cloudfront create-invalidation --distribution-id "$DIST" --paths "/*" >/dev/null
echo "Frontend: $URL"
echo "API:      $API"
