#!/bin/bash
set -e

LOCATION=$1
PROJECT_ID=$2
REPOSITORY_ID=$3
IMAGE_NAME=$4

echo "--- Cleaning up old images for $IMAGE_NAME in project $PROJECT_ID ---"

IMAGE_REPO="${LOCATION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY_ID}/${IMAGE_NAME}"
echo "Image repository: $IMAGE_REPO"

echo "Listing all images..."
ALL_IMAGES=$(gcloud artifacts docker images list "$IMAGE_REPO" --sort-by=~CREATE_TIME --limit=unlimited --format='get(version)')

if [ -z "$ALL_IMAGES" ]; then
  echo "No images found in the repository."
  exit 0
fi

echo "Found images:"
echo "$ALL_IMAGES"
IMAGE_COUNT=$(echo "$ALL_IMAGES" | wc -l)
echo "Total images found: $IMAGE_COUNT"
echo "--------------------"

IMAGES_TO_DELETE=$(echo "$ALL_IMAGES" | tail -n +6)

if [ -n "$IMAGES_TO_DELETE" ]; then
  echo "The following old images will be deleted:"
  echo "$IMAGES_TO_DELETE"
  
  echo "$IMAGES_TO_DELETE" | while read -r image_digest; do
    echo "Deleting image: $image_digest"
    gcloud artifacts docker images delete "${IMAGE_REPO}@$image_digest" --delete-tags --quiet
  done

  echo "--- Cleanup complete ---"
else
  echo "No old images to delete (found $IMAGE_COUNT images, keeping the last 5)."
fi
