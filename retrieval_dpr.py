from functools import lru_cache

from news import DPR


@lru_cache(maxsize=1)
def get_dpr() -> DPR:
    """Shared DPR retriever instance."""
    return DPR()
