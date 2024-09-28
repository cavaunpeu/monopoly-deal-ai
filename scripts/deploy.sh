#!/bin/bash

set -e

VERSION=${1:-"patch"}

# If version is a number, treat it as a full version
if [[ $VERSION =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    TAG="v$VERSION"
else
    # Otherwise, treat it as a semver increment
    # Get the highest version tag
    CURRENT_TAG=$(git tag -l | grep '^v[0-9]\+\.[0-9]\+\.[0-9]\+$' | sort -V | tail -1 || echo "v0.0.0")
    CURRENT_VERSION=${CURRENT_TAG#v}

    # Parse version
    IFS='.' read -r major minor patch <<< "$CURRENT_VERSION"

    case $VERSION in
        "major")
            major=$((major + 1))
            minor=0
            patch=0
            ;;
        "minor")
            minor=$((minor + 1))
            patch=0
            ;;
        "patch")
            patch=$((patch + 1))
            ;;
        *)
            echo "❌ Invalid version. Use: major, minor, patch, or x.y.z"
            exit 1
            ;;
    esac

    TAG="v$major.$minor.$patch"
fi

echo "🚀 Deploying version: $TAG"

# Check if tag already exists
if git tag -l | grep -q "^$TAG$"; then
    echo "❌ Tag $TAG already exists!"
    exit 1
fi

# Create and push tag
echo "📝 Creating tag: $TAG"
git tag $TAG

echo "📤 Pushing tag to trigger deployment..."
git push origin $TAG

echo "✅ Deployment triggered!"
echo "🔍 Check the Actions tab to monitor deployment progress"
echo "🌐 Your app will be available at the URLs shown in the workflow output"
