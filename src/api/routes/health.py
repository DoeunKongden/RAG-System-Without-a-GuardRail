import time

import httpx
from fastapi import APIRouter
from qdrant_client import QdrantClient

from src.config.settings import get_settings
from src.models import HealthCheckResponse

router = APIRouter()


@router.get("/health", response_model=HealthCheckResponse)
def health() -> HealthCheckResponse:
    settings = get_settings()
    start = time.monotonic()
    components: dict[str, bool] = {}

    # Ollama reachability
    try:
        with httpx.Client(timeout=5.0) as client:
            r = client.get(f"{str(settings.ollama_host).rstrip('/')}/api/tags")
            components["ollama"] = r.status_code == 200
    except Exception:
        components["ollama"] = False

    # Qdrant reachability
    try:
        qc = QdrantClient(url=str(settings.qdrant_url), timeout=5)
        qc.get_collections()
        components["qdrant"] = True
    except Exception:
        components["qdrant"] = False

    all_up = all(components.values())
    status = "healthy" if all_up else "degraded"

    return HealthCheckResponse(
        status=status,
        components=components,
        response_time_ms=(time.monotonic() - start) * 1000,
    )
