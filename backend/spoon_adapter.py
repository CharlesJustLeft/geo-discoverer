import os

def spoon_enabled():
    try:
        import spoon_ai  # type: ignore
    except Exception:
        return False
    return os.environ.get("SPOONOS_ENABLED", "").lower() in ("1", "true", "yes")