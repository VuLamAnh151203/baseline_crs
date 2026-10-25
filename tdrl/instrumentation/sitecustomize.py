"""Read-only allocator telemetry inside owned vLLM Python worker processes."""

import json
import os
import threading
import time
from pathlib import Path


def sample():
    import torch

    directory = Path(os.environ["TDRL_ALLOCATOR_TELEMETRY"])
    directory.mkdir(parents=True, exist_ok=True)
    while True:
        if torch.cuda.is_initialized():
            record = {
                "pid": os.getpid(),
                "timestamp": time.time(),
                "allocated_bytes": torch.cuda.memory_allocated(),
                "reserved_bytes": torch.cuda.memory_reserved(),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
            dest = directory / f"{os.getpid()}.json"
            temp = dest.with_suffix(".part")
            temp.write_text(json.dumps(record))
            temp.replace(dest)
        time.sleep(1)


if os.environ.get("TDRL_ALLOCATOR_TELEMETRY"):
    threading.Thread(target=sample, daemon=True).start()
