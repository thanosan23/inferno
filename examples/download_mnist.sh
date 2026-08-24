#!/usr/bin/env bash
set -euo pipefail

dest="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/mnist"
mkdir -p "$dest"

base_url="https://storage.googleapis.com/cvdf-datasets/mnist"
files=(
  train-images-idx3-ubyte.gz
  train-labels-idx1-ubyte.gz
  t10k-images-idx3-ubyte.gz
  t10k-labels-idx1-ubyte.gz
)

for f in "${files[@]}"; do
  if [ -f "$dest/$f" ]; then
    echo "already have $f"
    continue
  fi
  echo "downloading $f..."
  curl -fL -o "$dest/$f" "$base_url/$f"
done

echo "MNIST ready in $dest"
