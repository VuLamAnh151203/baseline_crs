#!/usr/bin/env python3
"""Dataset-specific entrypoint; intrinsic preflight implementation is shared."""

from preflight_difftod import main

if __name__ == "__main__":
    main(config="configs/difftod-durecdial-fit-v1.json", dataset="DuRecDial2.0")
