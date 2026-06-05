#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
# shellcheck source=.env
set -a; source .env; set +a
git pull origin main
# ── 1. Call pre-run.sh ────────────────────────────────────────────────────────
 bash pre-run.sh
# ── 2. Create a new git branch ────────────────────────────────────────────────
BRANCH="gtfs-gen/$(date +%Y-%m-%d)"
# Append a counter if the branch already exists
if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
  echo ""
  echo "Using branch $BRANCH"
  git checkout "$BRANCH"
else
  echo ""
  echo "Creating branch: $BRANCH"
  git checkout -b "$BRANCH"
fi
# ── 3. Call run.sh ────────────────────────────────────────────────────────────
echo "Running run script..."
i=0
while git diff --exit-code pmpml_gtfs_compat.zip
do
  echo "Running run script (ran $i times previously)"
  bash run.sh
  i=$((i+1))
  if git diff --exit-code pmpml_gtfs_compat.zip; then
    echo "Failed to run script to completion, sleeping for 15 minutes before retrying..."
    sleep 0.25h
  fi
done
# ── 4. Commit generated changes ───────────────────────────────────────────────
echo ""
echo "Committing generated output..."

git add --all

if git diff --cached --quiet; then
  echo "No changes detected after generator run — nothing to commit."
else
  COMMIT_MSG="regenerate GTFS dataset ($(date +%Y-%m-%d))"
  git commit -m "$COMMIT_MSG"
  echo ""
  echo "Committed on branch '$BRANCH': $COMMIT_MSG"
fi
# ── 5. Push changes and open merge request ────────────────────────────────────
echo "Pushing changes to remote..."
git push origin "$BRANCH"
REPO=$(git remote get-url origin | sed 's|.*github.com[:/]\(.*\)\.git|\1|;s|.*github.com[:/]\(.*\)|\1|')
SOURCE_URL="https://raw.githubusercontent.com/${REPO}/main/pmpml_gtfs.zip"
COMPAT_SRC_URL="https://raw.githubusercontent.com/${REPO}/main/pmpml_gtfs_compat.zip"
COMPARE_URL="https://raw.githubusercontent.com/${REPO}/${BRANCH}/pmpml_gtfs.zip"
COMPAT_COMPARE_URL="https://raw.githubusercontent.com/${REPO}/${BRANCH}/pmpml_gtfs_compat.zip"
COMPARE_LINK="https://gtfs.blrtransit.com/?source=${SOURCE_URL}&compare=${COMPARE_URL}"
COMPAT_COMPARE_LINK="https://gtfs.blrtransit.com/?source=${COMPAT_SRC_URL}&compare=${COMPAT_COMPARE_URL}"
gh pr create --base main --head "$BRANCH" --title "Automated GTFS dataset update ($BRANCH)" --body "Automated PR to merge regenerated GTFS dataset from branch \`$BRANCH\` into main. View and compare [here for gtfs.zip](${COMPARE_LINK}), [and here for gtfs_compat.zip](${COMPAT_COMPARE_LINK})." --repo "$REPO"
# ── 6. Change working tree back to main branch, delete $BRANCH locally ─────────
git checkout main
git branch -d "$BRANCH"
