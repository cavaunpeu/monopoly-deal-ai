#!/bin/bash
set -e

echo "🔄 Applying Terraform..."
terraform apply -auto-approve -var "project_id=monopoly-deal-agent"

echo "✅ Terraform Changes Applied!"