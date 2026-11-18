#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 SUBMIT_FILE [name=value ...]" >&2
  exit 2
fi
submit_file="$1"
shift
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
image_ref="$(<"${script_dir}/container-image.ref")"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
if [[ ! "$image_ref" =~ @sha256:[0-9a-f]{64}$ ]]; then
  echo "Container reference is not digest-pinned: $image_ref" >&2
  exit 2
fi
if [[ -n "$(git -C "$repo_root" status --porcelain)" ]]; then
  echo "Refusing submission: repository worktree is not clean." >&2
  git -C "$repo_root" status --short >&2
  exit 2
fi
git_commit="$(git -C "$repo_root" rev-parse HEAD)"
exec condor_submit "$@" "image=${image_ref}" "git_commit=${git_commit}" \
  "git_dirty=false" "$submit_file"
