#!/usr/bin/env python3
"""Audit released ConvPlan candidates and build a verified training-only EnPL bank."""

import argparse
import json
from pathlib import Path

from tdrl.baselines.enpl.bank import build_bank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--convplan-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-records", type=int)
    args = parser.parse_args()
    result = build_bank(
        args.root, args.convplan_zip, args.output, max_records=args.max_records
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
