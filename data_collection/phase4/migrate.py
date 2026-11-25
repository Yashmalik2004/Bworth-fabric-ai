"""Explicit, logged migration of a Phase 4 dataset to schema v2 / verification policy v2.

    python -m data_collection.phase4.migrate              # dry run: shows what would change, writes nothing
    python -m data_collection.phase4.migrate --apply      # backs up CSVs, adds v2 columns (empty), sets policy v2

Existing values are never modified. Fabrics verified with a single review become PENDING_VERIFICATION under
policy v2 until a second, independent reviewer accepts their evidence; the dry run lists them first.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from data_collection.phase4 import registry as R


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--apply", action="store_true", help="perform the migration (default: dry run)")
    a = ap.parse_args(argv)
    if not R.paths(a.root)["fabrics"].is_file():
        print("No Phase 4 dataset found at", R.paths(a.root)["root"])
        return 2
    rep = R.migrate(a.root, R.CURRENT_POLICY_VERSION, dry_run=not a.apply)
    print(json.dumps(rep, indent=2, default=str))
    print("APPLIED" if a.apply else "DRY RUN - nothing written. Re-run with --apply to migrate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
