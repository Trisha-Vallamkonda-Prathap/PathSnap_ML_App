"""PathSnap FastAPI backend (Member 4, runs on Laptop 1).

Start it with:  uvicorn pathsnap.service.api:app --host 0.0.0.0 --port 8000
"""
from .main import app  # noqa: F401  (re-exported so "pathsnap.service.api:app" works)
