import os

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader

from pipelines.churn import score_churn
from pipelines.impact_attribution import score_impact
from pipelines.interventions import score_interventions
from pipelines.resident_risk import score_residents
from pipelines.social_media import score_social_media

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


@app.post("/score/impact", dependencies=[Depends(verify_key)])
def trigger_impact_scoring():
    """Regenerate all donor impact statements."""
    try:
        count = score_impact.main()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"status": "impact statements generated", "count_scored": count}


@app.post("/score/residents", dependencies=[Depends(verify_key)])
def trigger_resident_scoring():
    """
    Score all active residents. Internal only — never exposed to donor-facing surfaces.
    Caller must have ML_SERVICE_API_KEY.
    """
    try:
        count = score_residents.main()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"status": "resident scoring complete", "count_scored": count}


@app.post("/score/interventions", dependencies=[Depends(verify_key)])
def trigger_intervention_scoring():
    """
    Generate per-resident intervention recommendations. Internal only — RLS restricted.
    Caller must have ML_SERVICE_API_KEY.
    """
    try:
        count = score_interventions.main()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"status": "intervention scoring complete", "count_scored": count}


@app.post("/score/social-media", dependencies=[Depends(verify_key)])
def trigger_social_media_scoring():
    """Generate posting recommendations for all platforms."""
    try:
        count = score_social_media.main()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"status": "social media recommendations generated", "count_written": count}
