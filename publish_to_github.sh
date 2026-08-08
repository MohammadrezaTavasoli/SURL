#!/usr/bin/env bash
set -euo pipefail

REPO_NAME="${1:-srul-generative-modeling}"
VISIBILITY="${2:-public}"
OWNER="${3:-MohammadrezaTavasoli}"

if ! command -v gh >/dev/null 2>&1; then
  echo "GitHub CLI (gh) is required: https://cli.github.com/" >&2
  exit 1
fi

if [[ "$VISIBILITY" != "public" && "$VISIBILITY" != "private" ]]; then
  echo "Visibility must be 'public' or 'private'." >&2
  exit 1
fi

gh auth status >/dev/null 2>&1 || gh auth login

gh repo create "$OWNER/$REPO_NAME" \
  "--$VISIBILITY" \
  --source=. \
  --remote=origin \
  --push

echo "Published: https://github.com/$OWNER/$REPO_NAME"
