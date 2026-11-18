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
if [[ ! "$image_ref" =~ @sha256:[0-9a-f]{64}$ ]]; then
  echo "Container reference is not digest-pinned: $image_ref" >&2
  exit 2
fi
exec condor_submit "$@" "image=${image_ref}" "$submit_file"
