#!/usr/bin/env bash
# Copy k8s/ to a temporary folder, fill in deployment values, and print the folder path.
# Usage: render_k8s.sh IMAGE_NAME IMAGE_TAG DNS_LABEL AZURE_REGION
set -euo pipefail

if [ "$#" -ne 4 ]; then
  echo "Usage: $0 IMAGE_NAME IMAGE_TAG DNS_LABEL AZURE_REGION" >&2
  exit 1
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output="$(mktemp -d)"
cp "$repo_root"/k8s/*.yaml "$output"/

sed -i \
  -e "s|__IMAGE_NAME__|$1|g" \
  -e "s|__IMAGE_TAG__|$2|g" \
  -e "s|__DNS_LABEL__|$3|g" \
  -e "s|__APP_DOMAIN__|$3.$4.cloudapp.azure.com|g" \
  "$output"/*.yaml

echo "$output"
