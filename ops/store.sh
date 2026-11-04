#!/usr/bin/env bash
# Sync the large data store (prices, filings, fundamentals, fitted models) with a GitHub release.
# Usage: ops/store.sh pull | push
# Needs GH_TOKEN with contents:write. The store is regenerable from public sources; git keeps
# only the small, auditable records (forecasts, ledger, scores, trades).
set -euo pipefail
TAG="datastore"
DIR="store"
case "${1:-}" in
  pull)
    mkdir -p "$DIR"
    if gh release view "$TAG" >/dev/null 2>&1; then
      rm -rf /tmp/store-dl && mkdir -p /tmp/store-dl
      if gh release download "$TAG" -D /tmp/store-dl -p 'store.tar*' --clobber; then
        cat /tmp/store-dl/store.tar.part-* > /tmp/store-dl/store.tar 2>/dev/null || true
        tar -xf /tmp/store-dl/store.tar -C "$DIR"
        echo "store restored: $(du -sh "$DIR" | cut -f1)"
      else
        echo "release exists but has no store yet"
      fi
    else
      echo "no datastore release yet (first run)"
    fi
    ;;
  push)
    if [ ! -d "$DIR" ] || [ -z "$(ls -A "$DIR")" ]; then echo "empty store; nothing to upload"; exit 0; fi
    gh release view "$TAG" >/dev/null 2>&1 || gh release create "$TAG" --title "Data store" --prerelease \
      --notes "Machine-maintained cache of public market data and fitted models used by the workflows. Regenerable; not part of the audited record."
    rm -rf /tmp/store-up && mkdir -p /tmp/store-up
    tar -cf /tmp/store-up/store.tar -C "$DIR" .
    # release assets are limited to 2 GiB each; split defensively at 1.5 GiB
    split -b 1500m -d -a 2 /tmp/store-up/store.tar /tmp/store-up/store.tar.part-
    rm /tmp/store-up/store.tar
    old=$(gh release view "$TAG" --json assets -q '.assets[].name' | grep '^store.tar' || true)
    for a in $old; do gh release delete-asset "$TAG" "$a" -y || true; done
    gh release upload "$TAG" /tmp/store-up/store.tar.part-* --clobber
    echo "store uploaded: $(du -sh "$DIR" | cut -f1)"
    ;;
  *) echo "usage: $0 pull|push"; exit 2 ;;
esac
