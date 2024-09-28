#!/bin/bash

set -e

PROJECT_ID=${1:-"your-project-id"}
REGION=${2:-"us-central1"}
DATABASE_INSTANCE="monopoly-deal-db"
DATABASE_NAME="mdeal"
DATABASE_USER="postgres"

echo "🚀 Setting up GCP resources for project: $PROJECT_ID"

# Set the project
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "📋 Enabling required APIs..."
gcloud services enable run.googleapis.com sqladmin.googleapis.com cloudbuild.googleapis.com containerregistry.googleapis.com --quiet

# Create Cloud SQL instance (if it doesn't exist)
echo "🗄️ Creating Cloud SQL instance..."
if gcloud sql instances describe $DATABASE_INSTANCE >/dev/null 2>&1; then
    echo "✅ Cloud SQL instance already exists"
else
    gcloud sql instances create $DATABASE_INSTANCE \
      --database-version=POSTGRES_15 \
      --tier=db-f1-micro \
      --region=$REGION \
      --storage-type=SSD \
      --storage-size=10GB \
      --storage-auto-increase \
      --backup \
      --no-assign-ip \
      --quiet
fi

# Create database (if it doesn't exist)
echo "📊 Creating database..."
if gcloud sql databases describe $DATABASE_NAME --instance=$DATABASE_INSTANCE >/dev/null 2>&1; then
    echo "✅ Database already exists"
else
    gcloud sql databases create $DATABASE_NAME --instance=$DATABASE_INSTANCE --quiet
fi

# Set database password (only if user doesn't exist)
echo "🔐 Setting database password..."
if gcloud sql users describe $DATABASE_USER --instance=$DATABASE_INSTANCE >/dev/null 2>&1; then
    echo "✅ Database user already exists, keeping existing password"
    # Get existing password (we can't retrieve it, so we'll generate a new one and show it)
    echo "⚠️  Note: If you need the password, check your GitHub Secrets or reset it manually"
else
    echo "🔑 Creating new database user with password..."
    DB_PASSWORD=$(openssl rand -base64 32)
    gcloud sql users create $DATABASE_USER \
      --instance=$DATABASE_INSTANCE \
      --password=$DB_PASSWORD \
      --quiet
fi

# Get connection info
echo "🔗 Getting connection info..."
DB_CONNECTION_NAME=$(gcloud sql instances describe $DATABASE_INSTANCE --format="value(connectionName)")
DB_PRIVATE_IP=$(gcloud sql instances describe $DATABASE_INSTANCE --format="value(ipAddresses[0].ipAddress)")

echo "✅ Setup complete!"
echo ""
echo "🔧 Next Steps:"
echo ""
echo "1. Database Password:"
if [ -n "$DB_PASSWORD" ]; then
    echo "   ✅ New user created with password: $DB_PASSWORD"
    echo "   💾 Save this password to a safe place now!"
    echo "   🔧 Set it on the database:"
    echo "      gcloud sql users set-password postgres --instance=$DATABASE_INSTANCE --password='$DB_PASSWORD'"
else
    echo "   ✅ Database user already exists (password unchanged)"
    echo "   🔧 Only reset if needed:"
    echo "      PASSWORD=\$(openssl rand -base64 32)"
    echo "      gcloud sql users set-password postgres --instance=$DATABASE_INSTANCE --password=\$PASSWORD"
fi
echo ""
echo "2. GitHub Secrets (Repository → Settings → Secrets):"
echo "   GCP_PROJECT_ID: $PROJECT_ID"
if [ -n "$DB_PASSWORD" ]; then
    echo "   DATABASE_URL: postgresql://$DATABASE_USER:$DB_PASSWORD@$DB_PRIVATE_IP:5432/$DATABASE_NAME"
else
    echo "   DATABASE_URL: postgresql://$DATABASE_USER:[EXISTING_PASSWORD]@$DB_PRIVATE_IP:5432/$DATABASE_NAME"
fi
echo "   FRONTEND_ORIGINS: https://monopoly-deal-frontend-[RANDOM].run.app"
echo "   ⚠️  [RANDOM] = placeholder - you'll get the real URL after first deployment"
echo ""
echo "3. Service Account Setup:"
echo "   ./scripts/setup-service-account.sh $PROJECT_ID"
echo ""
echo "4. Deploy & Get Real URLs:"
echo "   ./scripts/deploy.sh patch"
echo "   📋 After deployment, copy the real frontend URL and update FRONTEND_ORIGINS secret"
echo ""
echo "📋 Database Info:"
echo "   Connection: $DB_CONNECTION_NAME"
echo "   IP: $DB_PRIVATE_IP"
