"""
app/main.py - AI & Atmospheric Dispersion Physics Microservice
ForgeHack 2026 (ITHACKITS/AI-FORGEHACK-2026)
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List

from app.engine.simulator import SimulationRunner
from app.schemas.simulation import SimulationRequest
from app.impact.advisory import generate_emergency_advisory
from app.impact.gemini_advisory import enrich_advisory_with_gemini
from app.impact.validation import evaluate_event_validation

app = FastAPI(
    title="UdaraSiaga AI & Atmospheric Physics Microservice",
    description="Dedicated microservice for Eulerian-Lagrangian atmospheric dispersion simulation (Mastin 2009, Byram 1959) & Gemini LLM Disaster Advisory Synthesizer.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AdvisoryRequest(BaseModel):
    event_id: str
    hazard_type: str = "volcano"
    level: str = "Level IV (Awas)"
    exposed_schools: int = 0
    exposed_hospitals: int = 0
    exposed_population: int = 0
    plume_height_km: float = 5.0
    wind_direction_deg: float = 270.0
    wind_speed_ms: float = 8.5
    dominant_wind_cardinal: str = "Barat"
    sim_run_id: Optional[str] = None
    districts: Optional[List[str]] = None

@app.get("/")
def root():
    return {
        "service": "UdaraSiaga AI & Atmospheric Physics Engine",
        "competition": "ITHACKITS FORGEHACK 2026",
        "status": "online",
        "models": {
            "volcano_source": "Mastin (2009) Empirical Eruption Power Law (H = 2.0 * V^0.241)",
            "fire_source": "Byram (1959) Fireline Intensity & Plume Injection",
            "settling": "Stokes Settling with Cunningham Slip Correction",
            "land_mask": "Shapely 2.1 Vectorized Coastline Polygon Index",
            "llm_synthesizer": "Google Gemini 2.5 Flash + Deterministic BNPB/Kemenkes Fallback"
        }
    }

@app.get("/health")
def health():
    return {"status": "healthy", "service": "ai-physics-engine", "version": "2.0.0"}

@app.post("/api/v1/simulate")
def run_simulation(req: SimulationRequest):
    try:
        runner = SimulationRunner()
        result = runner.run(req)
        return {
            "status": "success",
            "sim_id": result.get("sim_id"),
            "event_id": req.event_id,
            "hazard_type": req.hazard_type,
            "summary": result.get("summary", {})
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/advisory")
def generate_advisory(req: AdvisoryRequest):
    try:
        base_advisory = generate_emergency_advisory(
            event_id=req.event_id,
            hazard_type=req.hazard_type,
            level=req.level,
            exposed_schools=req.exposed_schools,
            exposed_hospitals=req.exposed_hospitals,
            exposed_population=req.exposed_population,
            plume_height_km=req.plume_height_km,
            wind_direction_deg=req.wind_direction_deg,
            wind_speed_ms=req.wind_speed_ms,
            dominant_wind_cardinal=req.dominant_wind_cardinal,
            sim_run_id=req.sim_run_id,
            districts=req.districts
        )
        final_advisory = enrich_advisory_with_gemini(base_advisory)
        return {"status": "success", "advisory": final_advisory}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/validate")
def validate_scenario(event_id: str):
    try:
        val_res = evaluate_event_validation(event_id)
        return {"status": "success", "validation": val_res}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))