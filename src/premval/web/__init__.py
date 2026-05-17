"""Optional FastAPI web app for browsing ATLAS chains and (eventually) the
leaderboard. Requires the `[web]` extra (`pip install premval[web]`).
"""

from premval.web.app import Settings, create_app

__all__ = ["Settings", "create_app"]
