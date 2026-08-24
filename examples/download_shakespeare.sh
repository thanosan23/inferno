#!/usr/bin/env bash
set -euo pipefail

dest="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/shakespeare"
mkdir -p "$dest"

url="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
file="$dest/input.txt"

if [ -f "$file" ]; then
  echo "already have $file"
else
  echo "downloading tinyshakespeare..."
  curl -fL -o "$file" "$url"
fi

echo "shakespeare corpus ready at $file"
