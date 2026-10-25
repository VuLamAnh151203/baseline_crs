#!/usr/bin/env python3
"""DuRecDial entrypoint; native training/optimizer/resume code is shared."""

from train_difftod import main

if __name__ == "__main__":
    main(config="configs/difftod-durecdial-fit-v1.json", dataset="DuRecDial2.0")
