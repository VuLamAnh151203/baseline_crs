def ordered_generate(backend, requests):
    """Run independent generations with a fixed cap while preserving source order."""
    requests = list(requests)
    concurrency = min(
        len(requests),
        max(1, int(getattr(backend, "request_concurrency", 1))),
    )
    if concurrency > 1 and hasattr(backend, "generate_many"):
        return backend.generate_many(requests, max_workers=concurrency)
    return [backend.generate(*request) for request in requests]
