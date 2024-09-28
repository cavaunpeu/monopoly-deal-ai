#!/bin/bash
set -e

echo "🚀 Destroying GKE Autopilot Cluster..."

# Set variables
PROJECT_ID="monopoly-deal-agent"
CLUSTER_NAME="my-cluster-1"
REGION="us-central1"
SECRETS_BACKUP="secrets-backup.yaml"

echo "🔍 Verifying Cluster Exists..."
if gcloud container clusters describe "$CLUSTER_NAME" --region "$REGION" --project "$PROJECT_ID" > /dev/null 2>&1; then
    echo "📦 Backing up Kubernetes Secrets..."
    kubectl get secrets -A -o yaml > "$SECRETS_BACKUP"

    echo "🛑 Deleting GKE Cluster..."
    gcloud container clusters delete "$CLUSTER_NAME" --region "$REGION" --quiet
else
    echo "✅ No active cluster found. Skipping deletion."
fi

echo "🔄 Destroying Terraform-managed resources..."
terraform destroy -auto-approve

echo "🧹 Cleaning up local Terraform state..."
rm -rf .terraform terraform.tfstate terraform.tfstate.backup

echo "✅ Cluster teardown complete!"
echo "🔔 Secrets have been saved to $SECRETS_BACKUP. Use it to restore secrets when the new cluster is created."