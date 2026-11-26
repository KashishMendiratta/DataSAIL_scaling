"""Run all hard validation gates for a completed BindingDB C1e candidate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.bindingdb_c1e_validation import validate_c1e


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared_database", type=Path)
    parser.add_argument("assignment_database", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_c1e(
        args.prepared_database, args.assignment_database, args.output,
    ), indent=2))


if __name__ == "__main__":
    main()
