import os

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader

from pipelines.churn import score_churn

load_dotenv()

app = FastAPI(title="Lighthouse ML Service")

API_KEY        = os.getenv("ML_SERVICE_API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key")


def verify_key(key: str = Security(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/score/churn", dependencies=[Depends(verify_key)])
def trigger_scoring():
    """Trigger a batch scoring run. Called by .NET API on admin action."""
    try:
        count = score_churn.main()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"status": "scoring complete", "count_scored": count}
