#!/usr/bin/env bash
# Download and extract RUGD (~5 GB) into data/rugd (or the directory given as $1).
# Source: http://rugd.vision -- confirm the links there if a download fails.
set -euo pipefail
DEST="${1:-data/rugd}"
BASE="http://rugd.vision/data"
mkdir -p "$DEST"
cd "$DEST"
for f in RUGD_frames-with-annotations.zip RUGD_annotations.zip; do
  if [ ! -f "$f" ]; then
    curl -L --fail -o "$f" "$BASE/$f"
  fi
  unzip -q -n "$f"
done
echo "RUGD extracted to $(pwd)"
ls
