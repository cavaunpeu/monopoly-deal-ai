#!/bin/bash

set -e

PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
REGION="us-central1"
BACKEND_SERVICE="monopoly-deal-backend"
JOB_NAME="monopoly-deal-health-ping"

if [ -z "$PROJECT_ID" ]; then
    echo "❌ No active gcloud project. Run: gcloud config set project YOUR_PROJECT_ID"
    exit 1
fi

echo "🏓 Setting up health ping for $PROJECT_ID"

# Get backend URL dynamically
echo "📡 Getting backend URL..."
BACKEND_URL=$(gcloud run services describe $BACKEND_SERVICE --region=$REGION --format="value(status.url)" 2>/dev/null)

if [ -z "$BACKEND_URL" ]; then
    echo "❌ Backend service not found. Make sure it's deployed first."
    exit 1
fi

HEALTH_URL="$BACKEND_URL/health"
echo "✅ Backend URL: $BACKEND_URL"

# Test health endpoint
echo "🔍 Testing health endpoint..."
if curl -s -f "$HEALTH_URL" > /dev/null; then
    echo "✅ Health endpoint is working"
else
    echo "❌ Health endpoint not accessible. Deploy first."
    exit 1
fi

# Delete existing job if it exists
echo "🗑️  Removing existing job (if any)..."
gcloud scheduler jobs delete $JOB_NAME --location=$REGION --quiet 2>/dev/null || true

# Create new job
echo "⚡ Creating health ping job..."
gcloud scheduler jobs create http $JOB_NAME \
    --location=$REGION \
    --schedule="*/1 * * * *" \
    --uri="$HEALTH_URL" \
    --http-method=GET \
    --description="Keep Monopoly Deal AI warm" \
    --time-zone="UTC"

echo "🎉 Health ping setup complete!"
echo "   Job: $JOB_NAME"
echo "   URL: $HEALTH_URL"
echo "   Schedule: Every 1 minute"