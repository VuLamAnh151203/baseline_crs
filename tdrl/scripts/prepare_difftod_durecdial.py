#!/usr/bin/env python3
"""DuRecDial train/dev preparation; tokenizer/packing/resume implementation shared."""

from prepare_difftod_data import main

from tdrl.durecdial_diffusion_data import SERIALIZATION_VERSION, load_diffusion_split

if __name__ == "__main__":
    main(split_loader=load_diffusion_split, serialization=SERIALIZATION_VERSION)
