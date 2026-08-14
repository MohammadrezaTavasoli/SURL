#!/usr/bin/env bash
set -euo pipefail

REPO_NAME="${1:-srul-generative-modeling}"
VISIBILITY="${2:-public}"
OWNER="${3:-MohammadrezaTavasoli}"

if [[ "$VISIBILITY" != "public" && "$VISIBILITY" != "private" ]]; then
  echo "Visibility must be 'public' or 'private'." >&2
  exit 2
fi

if ! command -v git >/dev/null 2>&1; then
  echo "Git is required." >&2
  exit 2
fi

if ! command -v gh >/dev/null 2>&1; then
  echo "GitHub CLI is required: https://cli.github.com/" >&2
  exit 2
fi

if [[ ! -d .git ]]; then
  git init -b main
fi

if ! git rev-parse --verify HEAD >/dev/null 2>&1; then
  git add .
  git commit -m "Initial SRUL project release"
elif [[ -n "$(git status --porcelain)" ]]; then
  git add .
  git commit -m "Update SRUL project files"
fi

gh auth status >/dev/null 2>&1 || gh auth login

gh repo create "$OWNER/$REPO_NAME" \
  "--$VISIBILITY" \
  --source=. \
  --remote=origin \
  --push

printf 'Published: https://github.com/%s/%s\n' "$OWNER" "$REPO_NAME"
