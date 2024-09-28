#!/bin/bash
set -e

echo "🚀 Initializing Terraform..."
terraform init

echo "🔄 Applying Terraform to Create Cluster..."
terraform apply -auto-approve -var "project_id=monopoly-deal-agent"

echo "🔗 Connecting kubectl to the Cluster..."
gcloud container clusters get-credentials my-cluster-1 --region us-central1

echo "🔗 Creating the necessary k8s service accounts..."
kubectl create serviceaccount gcs-writer-ksa

echo "🔗 Annotating the k8s service account with the corresponding GCP service account..."
kubectl annotate serviceaccount gcs-writer-ksa iam.gke.io/gcp-service-account=$GCP_SERVICE_ACCOUNT

echo "🔑 Creating secrets..."
kubectl create secret generic wandb-api-key --from-literal=WANDB_API_KEY=$WANDB_API_KEY

echo "✅ Cluster Setup Complete!"