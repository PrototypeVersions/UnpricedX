#!/usr/bin/env bash
# Commit the auditable record produced by a workflow run and push it, retrying on races.
set -euo pipefail
MSG="${1:-update}"
git config user.name "unpriced-bot"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
for path in lab live site/data ops/done ops/requests; do
  if [ -e "$path" ]; then git add -A "$path"; fi
done
if git diff --cached --quiet; then echo "nothing to commit"; exit 0; fi
git commit -q -m "$MSG"
for i in 1 2 3 4 5; do
  if git push -q; then echo "pushed"; exit 0; fi
  echo "push rejected; rebasing (attempt $i)"
  git pull -q --rebase
done
echo "could not push" >&2; exit 1
