#!/bin/bash
set -e

# ==========================================
# PR Creation Helper Script
# ==========================================
# This script provides multiple methods to create PRs:
# 1. Try gh CLI (if installed)
# 2. Fallback to manual URL (always works)

REPO_OWNER="Shubhamf0073"
REPO_NAME="Margdrashti_models"
BASE_BRANCH="master"
CURRENT_BRANCH=$(git branch --show-current)

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}=================================================="
echo "CREATING PULL REQUEST"
echo -e "==================================================${NC}"
echo ""
echo "Branch: $CURRENT_BRANCH → $BASE_BRANCH"

# Get PR title and body
if [ -z "$1" ]; then
    # Use last commit message as title
    PR_TITLE=$(git log -1 --pretty=%B | head -1)
else
    PR_TITLE="$1"
fi

if [ -z "$2" ]; then
    # Use commit messages as body
    PR_BODY=$(git log $BASE_BRANCH..$CURRENT_BRANCH --pretty=format:"- %s")
else
    PR_BODY="$2"
fi

echo "Title: $PR_TITLE"
echo ""

# Method 1: Try gh CLI
if command -v gh &> /dev/null; then
    echo -e "${GREEN}✓ Using gh CLI${NC}"

    if gh pr create \
        --title "$PR_TITLE" \
        --body "$PR_BODY" \
        --base $BASE_BRANCH \
        --head $CURRENT_BRANCH; then
        echo -e "${GREEN}✅ Pull Request created successfully!${NC}"
        exit 0
    else
        echo -e "${YELLOW}⚠️  gh CLI failed, trying alternative...${NC}"
    fi
fi

# Method 2: Provide manual URL
echo -e "${YELLOW}=================================================="
echo "CREATE PR MANUALLY"
echo -e "==================================================${NC}"

# URL encode title
title_encoded=$(python3 -c "import sys; from urllib.parse import quote; print(quote('$PR_TITLE'))")
manual_url="https://github.com/$REPO_OWNER/$REPO_NAME/compare/$BASE_BRANCH...$CURRENT_BRANCH?expand=1&title=$title_encoded"

echo ""
echo -e "${BLUE}🔗 Click this link to create your PR:${NC}"
echo "$manual_url"
echo ""
echo "This will open GitHub with pre-filled information."
echo "Review the changes and click 'Create Pull Request'."
echo ""
echo -e "${GREEN}✅ URL generated successfully!${NC}"
exit 0
