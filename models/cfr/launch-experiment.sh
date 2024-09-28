#!/bin/bash
set -e

# Set variables
PROJECT_ID="monopoly-deal-agent"
IMAGE_NAME="my-experiment"
REGION="us-central1"
TAG=$(date +%Y%m%d-%H%M%S)  # Unique timestamp tag
JOB_NAME="experiment-job-$TAG"
FULL_IMAGE="gcr.io/$PROJECT_ID/$IMAGE_NAME:$TAG"
LATEST_IMAGE="gcr.io/$PROJECT_ID/$IMAGE_NAME:latest"
TEMP_YAML="temp-experiment-job.yaml"

# Capture command-line arguments for the experiment
EXPERIMENT_ARGS="$@"

echo "🚀 Building Docker image..."
# Abort if there are unstaged changes
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "Unstaged or uncommitted changes detected!"
    exit 1
fi
docker build --platform=linux/amd64 -f docker/cfr.Dockerfile -t $IMAGE_NAME .

if [ $? -ne 0 ]; then
    echo "❌ Local test failed. Fix errors before pushing."
    exit 1
fi

echo "🔄 Tagging image..."
docker tag $IMAGE_NAME $FULL_IMAGE
docker tag $IMAGE_NAME $LATEST_IMAGE  # Always overwrite latest

echo "📤 Pushing to GCR..."
gcloud auth configure-docker gcr.io --quiet > /dev/null 2>&1
docker push $LATEST_IMAGE  # Push latest

echo "✅ Successfully pushed: $FULL_IMAGE and $LATEST_IMAGE"

# Replace placeholders in YAML with actual values
echo "📄 Preparing Kubernetes Job YAML..."
GIT_COMMIT=$(git rev-parse HEAD)
EXPERIMENT_ARGS="\"--experiment-name\", \"$JOB_NAME\", \"--attempt-load-checkpoint\", \"--git-commit\", \"$GIT_COMMIT\""
for arg in "$@"; do
    EXPERIMENT_ARGS="$EXPERIMENT_ARGS, \"$arg\""
done

sed "s|{{ JOB_NAME }}|$JOB_NAME|g; s|{{ EXPERIMENT_ARGS }}|$EXPERIMENT_ARGS|g" models/cfr/experiment-job.yaml > $TEMP_YAML

# Apply the generated YAML file
echo "📤 Creating Kubernetes Job: $JOB_NAME..."
kubectl apply -f $TEMP_YAML

# Cleanup temp file
rm $TEMP_YAML

echo "🚀 Job Submitted! Run 'kubectl logs -f jobs/$JOB_NAME' to view logs."