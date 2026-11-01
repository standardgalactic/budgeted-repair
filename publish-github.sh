#!/usr/bin/env bash
set -euo pipefail

repo="standardgalactic/budgeted-repair"

command -v gh >/dev/null || {
    echo "GitHub CLI is required. Install it, then run gh auth login."
    exit 1
}
gh auth status

[[ "$(git rev-parse --show-toplevel)" == "$(pwd -P)" ]] || {
    echo "Run this from the budgeted-repair repository root."
    exit 1
}

branch=$(git symbolic-ref --short HEAD)

[[ -z "$(git diff --cached --name-only)" ]] || {
    echo "Commit or unstage existing staged changes first."
    exit 1
}

if git remote get-url origin >/dev/null 2>&1; then
    case "$(git remote get-url origin)" in
        "https://github.com/$repo"|"https://github.com/$repo.git"|"git@github.com:$repo.git") ;;
        *) echo "Origin points elsewhere; leaving it unchanged."; exit 1 ;;
    esac
fi

commit_files() {
    local message=$1
    shift
    if (( $# )); then
        git add -- "$@"
        if ! git diff --cached --quiet; then
            git commit -m "$message"
        fi
    fi
}

shopt -s nullglob
commit_files "Add bounded repair and convention experiments" ./*.py
commit_files "Add saved experimental results" results*/
commit_files "Add research manuscript and PDF" ./*.tex ./*.pdf
commit_files "Document experiments and publication workflow" README.md publish-github.sh

if gh repo view "$repo" >/dev/null 2>&1; then
    echo "Using existing repository: $repo"
else
    gh repo create "$repo" \
        --public \
        --description "Budget-constrained experiments in repair, conventions, inheritance, and turnover"
fi

if ! git remote get-url origin >/dev/null 2>&1; then
    git remote add origin "https://github.com/$repo.git"
fi

git push -u origin "$branch"

printf '\nRepository: https://github.com/%s\n' "$repo"
