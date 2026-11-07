#!/usr/bin/env bash
# Download the 14 MACS datasets from Zenodo (doi:10.5281/zenodo.23962155) and unpack them.
# Usage: bash scripts/download_datasets.sh [target folder]   (default: ../datasets/labeled_datasets)
set -euo pipefail
DEST="${1:-$(cd "$(dirname "$0")/.." && pwd)/../datasets/labeled_datasets}"
RECORD="https://zenodo.org/records/23962155/files"
mkdir -p "$DEST"
for name in membrane_datasets mito_datasets; do
    curl -fL -o "$DEST/$name.zip" "$RECORD/$name.zip?download=1"
    unzip -q -o "$DEST/$name.zip" -d "$DEST"
    rm "$DEST/$name.zip"
done
echo "Datasets in $DEST (set MACS_DATASETS_ROOT=$DEST if this is not ../datasets/labeled_datasets)"
