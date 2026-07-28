"""Validate a result directory supplied as the first argument."""

import sys

from chronocline.results import validate_result_directory

if __name__ == "__main__":
    errors = validate_result_directory(sys.argv[1])
    print("valid" if not errors else "\n".join(errors))
    raise SystemExit(bool(errors))
