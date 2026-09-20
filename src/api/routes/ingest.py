from fastapi import APIRouter

from src.ingestion.pipeline import run_ingestion
from src.models import IngestRequest, IngestResult

router = APIRouter()


@router.post("/ingest", response_model=IngestResult)
def ingest(request: IngestRequest) -> IngestResult:
    return run_ingestion(chunk_strategy=request.chunk_strategy)
