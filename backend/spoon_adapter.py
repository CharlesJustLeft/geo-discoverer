import os

def spoon_enabled():
    if os.environ.get("SPOONOS_ENABLED", "").lower() not in ("1", "true", "yes"):
        return False
    try:
        import spoon_ai_sdk  # type: ignore
        return True
    except Exception:
        pass
    try:
        import spoon  # type: ignore
        return True
    except Exception:
        pass
    return False

def spoon_version() -> str:
    try:
        import spoon_ai_sdk  # type: ignore
        return getattr(spoon_ai_sdk, "__version__", "unknown")
    except Exception:
        pass
    try:
        import spoon  # type: ignore
        return getattr(spoon, "__version__", "unknown")
    except Exception:
        pass
    return "not-installed"